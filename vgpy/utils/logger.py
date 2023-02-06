import os
import time
import logging.handlers
from logging.handlers import TimedRotatingFileHandler
import traceback
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

# LOG_NAME = 'test'
LOG_NAME = time.strftime("%Y-%m-%d", time.localtime())
LOG_DIR = str(Path(current_dir).parent.absolute()/'config'/'log')


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


logger = None

def create_logger(logName=LOG_NAME, logPath=LOG_DIR, to_msecs=True):
    global logger
    if logger is None:
        try:
            if not os.path.isdir(logPath):  
                os.makedirs(logPath)
                
            #Setting log
            logger = logging.getLogger(logName)
            logger.setLevel(logging.DEBUG)


            fileName = os.path.join(logPath, logName + '.log')
            __fHandler = TimedRotatingFileHandler(fileName, when="midnight", interval=1, encoding="utf-8")
            __fHandler.setLevel(logging.INFO)
            __fHandler.suffix = "%Y%m%d"

            fmt = '%(asctime)s [%(filename)s] %(levelname)-7s | %(message)s'

            if to_msecs:
                fmt = '%(asctime)s.%(msecs)03d [%(filename)s] %(levelname)-7s | %(message)s'

            __formatter= logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
            __fHandler.setFormatter(__formatter)
            
            logger.addHandler(__fHandler)

            # console handler
            __consoleHandler = logging.StreamHandler()
            __consoleHandler.setLevel(logging.DEBUG)
            __consoleHandler.setFormatter(__formatter)
            logger.addHandler(__consoleHandler)
        except:
            print('AUOLog init fail.')
            traceback.print_exc()

    return logger
        
   


def init_global_logger(logName, logPath='/log/', to_msecs=True):
    try:
        if not os.path.isdir(logPath):
            os.makedirs(logPath)
            
        #Setting log
        fileName = os.path.join(logPath, logName + '.log')
        __fHandler = TimedRotatingFileHandler(fileName, when="midnight", interval=1, encoding="utf-8")
        __fHandler.setLevel(logging.DEBUG)
        __fHandler.suffix = "%Y%m%d"

        fmt = '%(asctime)s [%(filename)s] %(levelname)-7s | %(message)s'

        if to_msecs:
            fmt = '%(asctime)s.%(msecs)03d [%(filename)s] %(levelname)-7s | %(message)s'

        __formatter= logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        __fHandler.setFormatter(__formatter)
        
        # console handler
        __consoleHandler = logging.StreamHandler()
        __consoleHandler.setLevel(logging.DEBUG)
        __consoleHandler.setFormatter(__formatter)

        logging.basicConfig(handlers=[__consoleHandler, __fHandler])
        logging.info('test')
    except:
        print('AUOLog init fail.')
        traceback.print_exc()

# def create_logger():
#     filename = time.strftime("%Y-%m-%d", time.localtime())+'.log'
#     logger = logging.getLogger()
#     logging.captureWarnings(False)
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s [%(filename)s - line:%(lineno)d] %(levelname)s: %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
    
#     # file handler
#     filepath = os.path.join(LOG_DIR, filename)
#     fileHandler = logging.FileHandler(filepath, 'a', 'utf-8')
#     fileHandler.setFormatter(formatter)
#     fileHandler.setLevel(logging.INFO)
#     logger.addHandler(fileHandler)

#     # console handler
#     consoleHandler = logging.StreamHandler()
#     consoleHandler.setFormatter(formatter)
#     logger.addHandler(consoleHandler)
#     return logger   

# if __name__ == '__main__':
#     logger = create_logger('wear')
#     logging.info("test!")
#     print('end')