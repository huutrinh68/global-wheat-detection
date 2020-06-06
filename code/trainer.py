from code.lib import *
from code.logger import log
from utils import AverageMeter

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


    def train(self, train_loader, validation_loader):
        for epoch in range(self.config.epochs):
            t =time.time()
            sum_loss = self.train_epoch()


    def train_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            self.optimizer.zero_grad()
            loss, _, _ = self.model(images, boxes, labels)
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()
            
        return summary_loss
            

