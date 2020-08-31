import configparser


class BertPredictionConfig(object):

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.bmp_config = config['Bert Match Prediction']

        self.bert_config_file = self.bmp_config['bert_config_file']
        self.vocab_file = self.bmp_config['vocab_file']
        self.init_checkpoint = self.bmp_config['init_checkpoint']
        self.max_seq_length = int(self.bmp_config['max_seq_length'])

        self.do_lower_case = bool(self.bmp_config['do_lower_case'])
        self.batch_size = int(self.bmp_config['batch_size'])
        self.use_tpu = bool(self.bmp_config['use_tpu'])
        self.tpu_name = self.bmp_config['tpu_name']
        self.tpu_zone = self.bmp_config['tpu_zone']
        self.gcp_project = self.bmp_config['gcp_project']
