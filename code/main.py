# add common library
from code.logger import logger, log
logger.setup('./logs', name='efficientDet')

from code.lib import *
from code.config import config
from code.utils import seed_everything, read_csv, kfold


if __name__ == '__main__':
    seed_everything(config.seed)

    # read csv
    data = read_csv(config.train_csv)

    # create stratify kfold
    kfold = kfold(data)
