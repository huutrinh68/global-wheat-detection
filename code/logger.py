import sys
import os
from datetime import datetime
import logging


# log multiline
class multilineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        save_msg = record.msg
        output = ''
        for line in save_msg.splitlines():
            record.msg = line
            output += super().format(record)  #+ '\n'
        record.ms = save_msg
        record.message = output

        return output


class Logger(object):

    def __init__(self):
        self.logger = logging.getLogger()        
        self.logger.setLevel(logging.INFO)

    def setup(self, dirname, name):

        os.makedirs(dirname, exist_ok=True)
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = f'{dirname}/{name}.log'
        # self.path = f'{dirname}/{name}_{now}.log'

        # log formater
        format='%(asctime)s- %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        if 1: # log one-line 
            formatter = logging.Formatter(format)
        else: #log multi-lines # not working perfect
            formatter = multilineFormatter(format)

        # write to file
        file_handler = logging.FileHandler(self.path, 'a')
        # file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # stream to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter) # comment out if doesnot to output time information
        self.logger.addHandler(console_handler)

        # log.info('')
        # log.info('--- %s ---' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # log.info(''.join(sys.argv))
        # log.info('logpath: %s' % self.path)

logger = Logger()
log = logger.logger

__all__ = ['logger', 'log']
