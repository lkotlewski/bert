import logging
import time
import traceback
from datetime import datetime
from os.path import join

from tensorflow import gfile

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
        files_names = [f for f in gfile.ListDirectory(dir_in) if not gfile.IsDirectory(join(dir_in, f))]
        for file_name in files_names:
            logging.info(file_name)
            file_path = join(dir_in, file_name)
            try:
                match_predictor.predict(dir_in, file_name, config.dir_result)
                gfile.Rename(file_path, join(config.dir_success, file_name))
            except Exception:
                logging.error(traceback.format_exc())
                gfile.Rename(file_path, join(config.dir_error, file_name))
        logging.info("job iteration finished")
        time.sleep(config.interval)


def configure_logging(config):
    log_format = '%(asctime)-15s %(message)s'
    now = datetime.now()
    log_filename = "{0}/{1}_{2}.log".format(config.dir_log, 'job_executor', now.strftime("%m_%d_%Y_%H-%M-%S"))
    logging.basicConfig(format=log_format, filename=log_filename, filemode='x', level=logging.DEBUG)


def make_needed_dirs(config):
    if not gfile.Exists(config.dir_in):
        gfile.MkDir(config.dir_in)
    if not gfile.Exists(config.dir_success):
        gfile.MkDir(config.dir_success)
    if not gfile.Exists(config.dir_error):
        gfile.MkDir(config.dir_error)
    if not gfile.Exists(config.dir_result):
        gfile.MkDir(config.dir_result)
    if not gfile.Exists(config.dir_log):
        gfile.MkDir(config.dir_log)


if __name__ == '__main__':
    execute_job()
