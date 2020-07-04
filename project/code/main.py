# add common library
from logger import logger, log
logger.setup('./logs', name='efficientDet')

from lib import *
from config import config
from dataset import WheatDataset, get_train_transforms, get_valid_transforms
from utils import seed_everything, read_csv, kfold
from trainer import Trainner, collate_fn

from efficientdet_master.effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from efficientdet_master.effdet.efficientdet import HeadNet


def get_net():
    config = get_efficientdet_config('tf_efficientdet_d7')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('./input/efficientdet/tf_efficientdet_d7-f05bf714.pth')
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 1024
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

def run_training():
    seed_everything(config.seed)
    device = torch.device(config.device)

    # read csv
    data_frame = read_csv(config.train_csv)

    # create stratify kfold
    df_folds = kfold(data_frame)

    # create dataset
    fold_number = 0
    train_dataset = WheatDataset(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        data_frame=data_frame,
        transforms=get_train_transforms(),
        test=False,
    )

    validation_dataset = WheatDataset(
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        data_frame=data_frame,
        transforms=get_valid_transforms(),
        test=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # model
    model  = get_net()
    model.to(device)

    # training
    trainer = Trainner(model=model, config=config)
    trainer.train(train_loader, val_loader)



if __name__ == '__main__':
    run_training()

