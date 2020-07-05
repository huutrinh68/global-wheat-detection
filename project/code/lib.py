import sys
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import warnings
warnings.simplefilter('ignore')
torch.backends.cudnn.benchmark=True
if torch.cuda.device_count() > 1:
  os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
