from lib import *
from logger import log
from utils import AverageMeter
from config import config
from tqdm import tqdm
class Trainner:
    def __init__(self, model, config, fold_number):
        self.config = config
        self.fold_number = fold_number
        self.epoch = 0

        self.best_summary_loss = 10**5
        self.model = model
        self.device = config.device
        self.checkpoint = config.checkpoint

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9, weight_decay=4e-5)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        log.info(f'Trainer prepared. Device is {self.device}')


    def train(self, train_loader, validation_loader):
        for epoch in range(self.config.n_epochs):
            t =time.time()
            summary_loss = self.train_epoch(train_loader)
            log.info(f'[RESULT]: Train. fold: {self.fold_number}, Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save_checkpoint(f'{self.checkpoint}/fold{self.fold_number}-last-checkpoint.pth')

            t = time.time()
            summary_loss = self.val_epoch(validation_loader)
            log.info(f'[RESULT]: Val. fold: {self.fold_number}, Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save_checkpoint(f'{self.checkpoint}/fold{self.fold_number}-best-checkpoint-{str(self.epoch).zfill(3)}epoch.pth')
                for path in sorted(glob(f'{self.checkpoint}/fold{self.fold_number}-best-checkpoint-*epoch.pth'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1


    def train_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for images, targets, image_ids in tqdm(train_loader):
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            # boxes = [b["bbox"].to(device) for b in targets]
            # labels = [l['cls'].to(device) for l in targets]
            targets = {}
            targets["bbox"] = boxes
            targets["cls"] = labels

            self.optimizer.zero_grad()
            loss = self.model(images, targets)
            loss['loss'].backward()

            summary_loss.update(loss['loss'].detach().item(), batch_size)
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss


    def val_epoch(self, val_loader):
        #self.model.eval()
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for images, targets, image_ids in tqdm(val_loader):
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                #
                target_res = {}

                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                #
                target_res["bbox"] = boxes
                target_res["cls"] = labels


                #loss, _, _ = self.model(images, boxes, labels)
                loss = self.model(images, target_res)
                summary_loss.update(loss['loss'].detach().item(), batch_size)

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


def collate_fn(batch):
    return tuple(zip(*batch))
