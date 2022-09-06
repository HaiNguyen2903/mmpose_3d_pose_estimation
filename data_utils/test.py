from xtcocotools.coco import COCO
from mmpose.datasets.datasets.custom_dataset.custom_2d_pose import Custom_2D_Dataset
import glob
import os
import os.path as osp
import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.zeros((3,2))

b[1] = (a[0] + a[2]) / 2
print(b)