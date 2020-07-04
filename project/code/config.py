from lib import *


class Config():
    train_csv = './input/global-wheat-detection/train.csv'
    train_imgs = './input/global-wheat-detection/train'
    checkpoint = './checkpoint'
    device = 'cuda:0'
    seed = 42
    lr = 0.0002
    n_epochs = 200
    batch_size = 2 
    num_workers = 4
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=2,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )


config = Config()
