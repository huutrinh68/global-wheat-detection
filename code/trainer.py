from code.lib import *
from code.logger import log
from utils import AverageMeter

class Trainner:
    def __init__(self, model, config):
        self.config = config
        self.epoch = 0

        self.best_summary_loss = 10**5
        self.model = model
        self.device = config.device
        self.checkpoint = config.checkpoint

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        log.info(f'Trainer prepared. Device is {self.device}')


    def train(self, train_loader, validation_loader):
        for epoch in range(self.config.n_epochs):
            t =time.time()
            summary_loss = self.train_epoch(train_loader)
            log.info(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save_checkpoint(f'{self.checkpoint}/last-checkpoint.pth')

            t = time.time()
            summary_loss = self.val_epoch(validation_loader)
            log.info(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save_checkpoint(f'{self.checkpoint}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.pth')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.pth'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1




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


    def val_epoch(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                loss, _, _ = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss
    

    def save_checkpoint(self, path):
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
