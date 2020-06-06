from code.lib import *
from code.logger import log

class Trainner:
    def __init__(self, model, config):
        self.config = config
        self.epoch = 0

        self.best_loss = 0
        self.model = model
        self.device = config.device

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        log.info(f'Trainer prepared. Device is {self.device}')
