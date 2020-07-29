from lib import *

class Config():
    train_csv = './input/global-wheat-detection/train.csv'
    train_imgs = './input/global-wheat-detection/train'
    #checkpoint = './checkpoint-d5-master'
    checkpoint = './checkpoint-d5-namdt-sgd1'
    # gpu_ids = [0, 1, 2, 3] # change to gpus you want to use
    gpu_ids = [0]
    if len(gpu_ids) > 1:
        device = torch.device(f'cuda:{gpu_ids}')
    else:
        device = torch.device('cuda:0')
    use_pretrained = False
    seed = 42
    #lr = 0.0004
    lr = 0.005
    n_epochs = 200
    batch_size = 2
    accumulate = 8
    num_workers = 3
    step_scheduler = True
    validation_scheduler = False
    #SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingLR
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
