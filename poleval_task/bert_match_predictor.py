import os

import tensorflow as tf

import modeling
import tokenization
from match_prediction_config import MatchPredictionConfig
from run_classifier import PlwiProcessor, PaddingInputExample, file_based_convert_examples_to_features, \
    file_based_input_fn_builder, model_fn_builder


class BertMatchPredictor:

    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.config = MatchPredictionConfig()

        bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.config.vocab_file, do_lower_case=self.config.do_lower_case)

        self.processor = PlwiProcessor()
        self.label_list = self.processor.get_labels()

        tpu_cluster_resolver = None
        if self.config.use_tpu and self.config.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.config.tpu_name, zone=self.config.tpu_zone, project=self.config.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir="out",
            save_checkpoints_steps=False,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=1000,
                num_shards=8,
                per_host_input_for_training=is_per_host))

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self.label_list),
            init_checkpoint=self.config.init_checkpoint,
            learning_rate=0,
            num_train_steps=0,
            num_warmup_steps=0,
            use_tpu=self.config.use_tpu,
            use_one_hot_embeddings=self.config.use_tpu)

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.config.batch_size,
            eval_batch_size=self.config.batch_size,
            predict_batch_size=self.config.batch_size)

    def predict(self, dir_in, filename, dir_out, logger=None):
        predict_examples = self.processor.get_examples_from(os.path.join(dir_in, filename))
        num_actual_predict_examples = len(predict_examples)
        if self.config.use_tpu:
            while len(predict_examples) % self.config.batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(dir_out, "{0}.tf_record".format(self.get_filename_without_extension(filename)))
        file_based_convert_examples_to_features(predict_examples, self.label_list,
                                                self.config.max_seq_length, self.tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", self.config.batch_size)

        predict_drop_remainder = True if self.config.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.config.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = self.estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(dir_out,
                                           "{0}_result.tsv".format(self.get_filename_without_extension(filename)))
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

    @staticmethod
    def get_filename_without_extension(filename):
        return os.path.splitext(filename)[0]


if __name__ == '__main__':
    bert = BertMatchPredictor()
    bert.predict("C:/Users/Łukasz/Desktop/programowanie/bert/PlWi/test-min.tsv", "results")
