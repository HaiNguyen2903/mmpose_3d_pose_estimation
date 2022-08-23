import os
import os.path as osp
import cv2
from moviepy.editor import *

def mkdir_if_missing(root):
    if not osp.exists(root):
        print(f'mkdir {root}')
        os.makedirs(root)


def trim_video(video_path, save_path):
    # loading video dsa gfg intro video
    clip = VideoFileClip(video_path)
        
    # getting only first 5 seconds
    clip = clip.subclip(0, 30)

    # saving the clip
    clip.write_videofile(save_path)

def extract_frames(video_path, save_dir):
    cap= cv2.VideoCapture(video_path)
    i=0

    mkdir_if_missing(save_dir)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(osp.join(save_dir, '{:05}'.format(i) + '.jpg'), frame)
        print(f'Extracting frame {i}')
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = '/home/ducanh/hain/dataset/175_30s.mp4'
    save_dir = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/posetrack18_format/images'

    extract_frames(video_path, save_dir)
