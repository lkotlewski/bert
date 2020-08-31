import csv
import logging
import os
import string

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from keyed_vectors_prediction_config import KeyedVectorsPredictionConfig


class KeyedVectorsFormatPredictor:

    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.config = KeyedVectorsPredictionConfig()
        logging.info("loading keyed vectors file")
        if self.config.bin_file:
            self.word2vec = KeyedVectors.load(self.config.keyed_vectors_model)
        else:
            self.word2vec = KeyedVectors.load_word2vec_format(self.config.keyed_vectors_model)
        logging.info("keyed vectors loaded")

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _create_context_pairs(cls, lines):
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line[4]
            text_b = line[5]
            examples.append(
                ContextPair(text_a=text_a, text_b=text_b))
        return examples

    def predict(self, dir_in, filename, dir_out):
        context_pairs = self._create_context_pairs(
            self._read_tsv(os.path.join(dir_in, filename)))

        result = []
        for context_pair in context_pairs:

            words_a = self.get_words(context_pair.text_a)
            words_b = self.get_words(context_pair.text_b)
            similarity = float(0)
            if len(words_a) > 0 and len(words_b) > 0:
                text_a_embedding = self.get_mean_vector(words_a)
                text_b_embedding = self.get_mean_vector(words_b)
                similarity = cosine_similarity([text_a_embedding], [text_b_embedding])[0][0]
            result.append(similarity)

        output_predict_file = os.path.join(dir_out,
                                           "{0}_result.tsv".format(self.get_filename_without_extension(filename)))

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for (i, similarity) in enumerate(result):
                output_line = str(similarity) + "\n"
                writer.write(output_line)

    @classmethod
    def get_words(cls, text):
        return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

    def get_mean_vector(self, words):
        words = [word for word in words if word in self.word2vec.vocab]
        if len(words) >= 1:
            return np.mean(self.word2vec[words], axis=0)
        else:
            return []

    @classmethod
    def get_filename_without_extension(cls, filename):
        return os.path.splitext(filename)[0]


def _read_tsv(input_file, quotechar=None):
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def _create_context_pairs(lines):
    examples = []
    for (i, line) in enumerate(lines):
        text_a = line[4]
        text_b = line[5]
        examples.append(
            ContextPair(text_a=text_a, text_b=text_b))
    return examples


def get_mean_vector(self, words):
    words = [word for word in words if word in self.word2vec.vocab]
    if len(words) >= 1:
        return np.mean(self.word2vec[words], axis=0)
    else:
        return []


class ContextPair(object):
    def __init__(self, text_a, text_b):
        self.text_a = text_a
        self.text_b = text_b
