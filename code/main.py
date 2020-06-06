# add common library
from code.logger import logger, log
logger.setup('./logs', name='efficientDet')

from code.lib import *
from code.config import config
from code.dataset import WheatDataset, get_train_transforms, get_valid_transforms
from code.utils import seed_everything, read_csv, kfold
from code.trainer import Trainner


if __name__ == '__main__':
    seed_everything(config.seed)

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

    image, target, image_id = train_dataset[1]
    boxes = target['boxes'].cpu().numpy().astype(np.int32)
    numpy_image = image.permute(1,2,0).cpu().numpy()

    for box in boxes:
        cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 2)
    cv2.imshow('img', numpy_image)
    cv2.waitKey(0)

    # test with dummy model
    import torchvision.models as models
    trainner = Trainner(models.resnet18(), config)