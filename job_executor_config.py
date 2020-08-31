import configparser


class JobExecutorConfig(object):

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.mp_config = config['Job Executor']

        self.dir_in = self.mp_config['dir_in']
        self.dir_success = self.mp_config['dir_success']
        self.dir_error = self.mp_config['dir_error']
        self.dir_result = self.mp_config['dir_result']
        self.dir_log = self.mp_config['dir_log']
        self.interval = float(self.mp_config['interval'])
        self.prediction = self.mp_config['prediction']

