import configparser


class KeyedVectorsPredictionConfig(object):

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.kvmp_config = config['Keyed Vectors Match Predictor']

        self.keyed_vectors_model = self.kvmp_config['keyed_vectors_model']
        self.bin_file = bool(self.kvmp_config['bin_file'])

