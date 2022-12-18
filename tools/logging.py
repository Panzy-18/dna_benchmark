import logging
import bmtrain as bmt
from .global_var import GLOBAL_VAR
import os

def get_logger(name):
    if os.environ.get('LOCAL_RANK') is None:
        return None
    
    logger = logging.getLogger(name)
    
    if name in GLOBAL_VAR._logger:
        return logger
    
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    
    if int(os.environ['LOCAL_RANK']) == 0 and not GLOBAL_VAR._args.no_log:
        file_handler = logging.FileHandler(GLOBAL_VAR._args.log_file, 'a+')
        handlers.append(file_handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    
    if int(os.environ['LOCAL_RANK']) == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    
    GLOBAL_VAR._logger[name] = True
    return logger

def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        bmt.print_rank(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
