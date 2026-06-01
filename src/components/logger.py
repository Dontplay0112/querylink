import logging
import os
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

from components.config import MyConfig

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(f"\r\033[K{msg}")
            self.flush()
        except Exception:
            self.handleError(record)

def init_logging(config: MyConfig):
    log_dir = config.exp.output_path
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    root_logger = logging.getLogger()
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[ %(levelname)s ] \033[38;2;57;197;187m[ %(asctime)s | %(processName)s | %(filename)s:%(lineno)d ]\033[0m %(message)s'
    )
    
    fh = RotatingFileHandler(os.path.join(log_dir, "app.log"), maxBytes=100*1024*1024, backupCount=3)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    
    ch = TqdmLoggingHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)