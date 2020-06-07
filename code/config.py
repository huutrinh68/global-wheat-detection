from code.lib import *


class Config():
    train_csv = './input/global-wheat-detection/train.csv'
    train_imgs = './input/global-wheat-detection/train'
    checkpoint = './checkpoint'
    device = 'cuda:0'
    seed = 42
    lr = 0.001
    n_epochs = 20
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )


config = Config()
