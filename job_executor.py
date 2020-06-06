import logging
import time
import traceback
from os import listdir, replace, mkdir
from os.path import isfile, join, exists
from datetime import datetime

from bert_match_predictor import BertMatchPredictor
from job_executor_config import JobExecutorConfig


def execute_job():
    config = JobExecutorConfig()
    make_needed_dirs(config)
    configure_logging(config)
    match_predictor = BertMatchPredictor()
    while True:
        logging.info("job iteration started")
        dir_in = config.dir_in
        files_names = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
        for file_name in files_names:
            logging.info(file_name)
            file_path = join(dir_in, file_name)
            try:
                match_predictor.predict(dir_in, file_name, config.dir_result)
                replace(file_path, join(config.dir_success, file_name))
            except Exception:
                logging.error(traceback.format_exc())
                replace(file_path, join(config.dir_error, file_name))
        logging.info("job iteration finished")
        time.sleep(config.interval)


def configure_logging(config):
    log_format = '%(asctime)-15s %(message)s'
    now = datetime.now()
    log_filename = "{0}/{1}_{2}.log".format(config.dir_log, 'job_executor', now.strftime("%m_%d_%Y_%H-%M-%S"))
    logging.basicConfig(format=log_format, filename=log_filename, filemode='x', level=logging.DEBUG)


def make_needed_dirs(config):
    if not exists(config.dir_in):
        mkdir(config.dir_in)
    if not exists(config.dir_success):
        mkdir(config.dir_success)
    if not exists(config.dir_error):
        mkdir(config.dir_error)
    if not exists(config.dir_result):
        mkdir(config.dir_result)
    if not exists(config.dir_log):
        mkdir(config.dir_log)


if __name__ == '__main__':
    execute_job()
