# Import everything needed to edit video clips
from moviepy.editor import *
import glob
import os.path as osp
import os

video_path = '/home/ducanh/hain/dataset/175.mp4'
save_path = '/home/ducanh/hain/dataset/175_5s.mp4'

# loading video dsa gfg intro video
clip = VideoFileClip(video_path)
	
# getting only first 5 seconds
clip = clip.subclip(0, 5)

# saving the clip
# clip.write_videofile(save_path)

# S1_Seq1_Cam0_000001.jpg

data_root = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/mpi_3dhf_format/images'

img_paths = glob.glob(osp.join(data_root, '*.jpg'))

for path in img_paths:
    img_name = osp.basename(path).split('.')[0]
    sub, seq, cam, id = img_name.split('_')
    new_name = f"{sub}_Seq1_{cam}_{id}.jpg"
    os.rename(path, osp.join(data_root, new_name))