from xtcocotools.coco import COCO
from mmpose.datasets.datasets.custom_dataset.custom_2d_pose import Custom_2D_Dataset
import glob
import os
import os.path as osp

ann_path = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/posetrack18_format/2dtrack_train.json'

data_dir = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/posetrack18_format/images/val'
imgs = glob.glob(osp.join(data_dir, '*.jpg'))

for img in imgs:
    name = osp.basename(img).split('.')[0]
    new_name = "{:06}".format(int(name))
    os.rename(img, osp.join(data_dir, new_name + '.jpg'))