import configparser


class MatchPredictionConfig(object):

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.mp_config = config['Match Prediction']

        self.bert_config_file = self.mp_config['bert_config_file']
        self.vocab_file = self.mp_config['vocab_file']
        self.init_checkpoint = self.mp_config['init_checkpoint']
        self.max_seq_length = int(self.mp_config['max_seq_length'])

        self.do_lower_case = bool(self.mp_config['do_lower_case'])
        self.batch_size = int(self.mp_config['batch_size'])
        self.use_tpu = bool(self.mp_config['use_tpu'])
        self.tpu_name = self.mp_config['tpu_name']
        self.tpu_zone = self.mp_config['tpu_zone']
        self.gcp_project = self.mp_config['gcp_project']
