from lib import *
from config import config


def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=1024, width=1024, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            #A.Flip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=1024, width=1024, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


# use cutmix method
# https://paperswithcode.com/paper/cutmix-regularization-strategy-to-train
class WheatDataset(Dataset):
    def __init__(self, data_frame, image_ids, transforms=None, test=False):
        super(WheatDataset, self).__init__()
        self.image_ids = image_ids
        self.data_frame = data_frame
        self.transforms = transforms
        self.test = test

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if self.test or random.random() > 0.5:
            image, boxes = self.load_image_and_boxes(idx)
        elif random.random() > 0.25:
            image, boxes = self.load_cutmix_image_and_boxes(idx)
        else:
            image, boxes = self.load_image_and_bboxes_with_cutmix(idx)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if self.transforms:
            for i in range(10):

                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': target['labels']
                })

                #assert len(sample['bboxes']) == labels.shape[0], 'not equal!'

                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    #yxyx: be warning
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]
                    target['labels'] = torch.stack(sample['labels']) # <--- add this!
                    break

        return image, target, image_id


    def __len__(self):
        return self.image_ids.shape[0]


    def load_image_and_boxes(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(config.train_imgs, image_id+'.jpg')

        image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # normalization
        image /= 255.0

        # get x, y, w, h information
        boxes = self.data_frame[self.data_frame['image_id'] == image_id][['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # convert w -> x
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # convert h -> y
        return image, boxes


    def load_cutmix_image_and_boxes(self, idx, imsize=1024):
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize*0.25, imsize*0.75)) for _ in range(2)]
        idxs = [idx] + [random.randint(0, self.image_ids.shape[0] -1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, idx in enumerate(idxs):
            image, boxes = self.load_image_and_boxes(idx)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]

        return result_image, result_boxes

    def load_image_and_bboxes_with_cutmix(self, idx):
        image, bboxes = self.load_image_and_boxes(idx)
        image_to_be_mixed, bboxes_to_be_mixed = self.load_image_and_boxes(
            random.randint(0, self.image_ids.shape[0] - 1))

        image_size = image.shape[0]
        cutoff_x1, cutoff_y1 = [int(random.uniform(image_size * 0.0, image_size * 0.49)) for _ in range(2)]
        cutoff_x2, cutoff_y2 = [int(random.uniform(image_size * 0.5, image_size * 1.0)) for _ in range(2)]

        image_cutmix = image.copy()
        image_cutmix[cutoff_y1:cutoff_y2, cutoff_x1:cutoff_x2] = image_to_be_mixed[cutoff_y1:cutoff_y2,
                                                                 cutoff_x1:cutoff_x2]

        # Begin preparing bboxes_cutmix.
        # Case 1. Bounding boxes not intersect with cut off patch.
        bboxes_not_intersect = bboxes[np.concatenate((np.where(bboxes[:, 0] > cutoff_x2),
                                                      np.where(bboxes[:, 2] < cutoff_x1),
                                                      np.where(bboxes[:, 1] > cutoff_y2),
                                                      np.where(bboxes[:, 3] < cutoff_y1)), axis=None)]

        # Case 2. Bounding boxes intersect with cut off patch.
        bboxes_intersect = bboxes.copy()

        top_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                 (bboxes[:, 2] > cutoff_x1) &
                                 (bboxes[:, 1] < cutoff_y2) &
                                 (bboxes[:, 3] > cutoff_y2))
        right_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                   (bboxes[:, 2] > cutoff_x2) &
                                   (bboxes[:, 1] < cutoff_y2) &
                                   (bboxes[:, 3] > cutoff_y1))
        bottom_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                    (bboxes[:, 2] > cutoff_x1) &
                                    (bboxes[:, 1] < cutoff_y1) &
                                    (bboxes[:, 3] > cutoff_y1))
        left_intersect = np.where((bboxes[:, 0] < cutoff_x1) &
                                  (bboxes[:, 2] > cutoff_x1) &
                                  (bboxes[:, 1] < cutoff_y2) &
                                  (bboxes[:, 3] > cutoff_y1))

        # Remove redundant indices. e.g. a bbox which intersects in both right and top.
        right_intersect = np.setdiff1d(right_intersect, top_intersect)
        right_intersect = np.setdiff1d(right_intersect, bottom_intersect)
        right_intersect = np.setdiff1d(right_intersect, left_intersect)
        bottom_intersect = np.setdiff1d(bottom_intersect, top_intersect)
        bottom_intersect = np.setdiff1d(bottom_intersect, left_intersect)
        left_intersect = np.setdiff1d(left_intersect, top_intersect)

        bboxes_intersect[:, 1][top_intersect] = cutoff_y2
        bboxes_intersect[:, 0][right_intersect] = cutoff_x2
        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1
        bboxes_intersect[:, 2][left_intersect] = cutoff_x1

        bboxes_intersect[:, 1][top_intersect] = cutoff_y2
        bboxes_intersect[:, 0][right_intersect] = cutoff_x2
        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1
        bboxes_intersect[:, 2][left_intersect] = cutoff_x1

        bboxes_intersect = bboxes_intersect[np.concatenate((top_intersect,
                                                            right_intersect,
                                                            bottom_intersect,
                                                            left_intersect), axis=None)]

        # Case 3. Bounding boxes inside cut off patch.
        bboxes_to_be_mixed[:, [0, 2]] = bboxes_to_be_mixed[:, [0, 2]].clip(min=cutoff_x1, max=cutoff_x2)
        bboxes_to_be_mixed[:, [1, 3]] = bboxes_to_be_mixed[:, [1, 3]].clip(min=cutoff_y1, max=cutoff_y2)

        # Integrate all those three cases.
        bboxes_cutmix = np.vstack((bboxes_not_intersect, bboxes_intersect, bboxes_to_be_mixed)).astype(int)
        bboxes_cutmix = bboxes_cutmix[np.where((bboxes_cutmix[:, 2] - bboxes_cutmix[:, 0]) \
                                               * (bboxes_cutmix[:, 3] - bboxes_cutmix[:, 1]) > 500)]
        # End preparing bboxes_cutmix.

        return image_cutmix, bboxes_cutmix

    def load_mixup_image_and_boxes(self, idx):
      image, boxes = self.load_image_and_boxes(idx)
      r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
      return (image+r_image)/2, np.vstack((boxes, r_boxes)).astype(np.int32)
