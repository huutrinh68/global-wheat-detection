# add common library
from logger import logger, log
logger.setup('./logs', name='efficientDet')

from lib import *
from config import config
from dataset import WheatDataset, get_train_transforms, get_valid_transforms
from utils import seed_everything, read_csv, kfold
from trainer import Trainner, collate_fn


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

    # # model
    # model.to(device)

    # # training
    # trainer = Trainner(model=model, config=config)
    # trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    run_training()

