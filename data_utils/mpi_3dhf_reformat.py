import os
import os.path as osp
import glob
from data_utils.utils import *
import json
import numpy as np
import pickle

def get_pose_stats(kps):
    """Get statistic information `mean` and `std` of pose data.

    Args:
        kps (ndarray): keypoints in shape [..., K, C] where K and C is
            the keypoint category number and dimension.
    Returns:
        mean (ndarray): [K, C]
    """
    assert kps.ndim > 2
    K, C = kps.shape[-2:]
    kps = kps.reshape(-1, K, C)
    # mean = kps.mean(axis=0, where=kps>0)
    # std = kps.std(axis=0, where=kps>0)
    mean = kps.mean(axis=0)
    std = kps.std(axis=0)
    return mean, std


def get_annotations(joints_2d, joints_3d, scale_factor=1.2, train_img_size=(608, 1080)):
    """Get annotations, including centers, scales, joints_2d and joints_3d.

    Args:
        joints_2d: 2D joint coordinates in shape [N, K, 2], where N is the
            frame number, K is the joint number.
        joints_3d: 3D joint coordinates in shape [N, K, 3], where N is the
            frame number, K is the joint number.
        scale_factor: Scale factor of bounding box. Default: 1.2.
    Returns:
        centers (ndarray): [N, 2]
        scales (ndarray): [N,]
        joints_2d (ndarray): [N, K, 3]
        joints_3d (ndarray): [N, K, 4]
    """
    # calculate joint visibility
    visibility = (joints_2d[:, :, 0] >= 0) * \
                 (joints_2d[:, :, 0] < train_img_size[0]) * \
                 (joints_2d[:, :, 1] >= 0) * \
                 (joints_2d[:, :, 1] < train_img_size[1])
    visibility = np.array(visibility, dtype=np.float32)[:, :, None]
    joints_2d = np.concatenate([joints_2d, visibility], axis=-1)
    joints_3d = np.concatenate([joints_3d, visibility], axis=-1)

    # calculate bounding boxes
    bboxes = np.stack([
        np.min(joints_2d[:, :, 0], axis=1),
        np.min(joints_2d[:, :, 1], axis=1),
        np.max(joints_2d[:, :, 0], axis=1),
        np.max(joints_2d[:, :, 1], axis=1)
    ],
                      axis=1)
    centers = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2,
                        (bboxes[:, 1] + bboxes[:, 3]) / 2],
                       axis=1)
    scales = scale_factor * np.max(bboxes[:, 2:] - bboxes[:, :2], axis=1) / 200

    return centers, scales, joints_2d, joints_3d

if __name__ == '__main__':
    save_dir = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/mpi_3dhf_format'
    result_2d = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/175_30s_2d_anno_new.json'
    result_3d = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/vis_results/vis_175_30s.npy'
    out_train_file = osp.join(save_dir, 'annotations/train.npz')
    out_val_file = osp.join(save_dir, 'annotations/val.npz')
    train_img_size = (608, 1080)

    f = open(result_2d)
    data = json.load(f)

    mkdir_if_missing(osp.join(save_dir, 'annotations'))
    mkdir_if_missing(osp.join(save_dir, 'images'))

    '''
    create train.npz
    '''
    imgnames = ['S1_Seq1_Cam0_{:06}.jpg'.format(i) for i in range(900)]
    
    joints_2d = [frame[0]['keypoints'] for frame in data]
    # joints_2d = np.array([np.array(joint_2d)[:, :2] for joint_2d in joints_2d])

    '''
    reorder and create new kpts to map with 3d kpts
    2d index -> 3d index 
        0: 'nose',              # head top
        1: 'left_eye',          # neck
        2: 'right_eye',         # right shoulder 
        3: 'left_ear',          # right elbow
        4: 'right_ear',         # right wrist 
        5: 'left_shoulder',     # left shoulder
        6: 'right_shoulder',    # left elbow
        7: 'left_elbow',        # left wrist 
        8: 'right_elbow',       # right hip
        9: 'left_wrist',        # right knee
        10: 'right_wrist',      # right ankle
        11: 'left_hip',         # left hip
        12: 'right_hip',        # left knee
        13: 'left_knee',        # left ankle
        14: 'right_knee',       # root
        15: 'left_ankle',       # spine
        16: 'right_ankle'       # center head ?

    left ear right ear -> remove
    left eye right eye -> replace by head top
    left shoulder right shoulder -> create neck
    left hip right hip -> create root
    root neck -> create spine

    1 -> head top
    2 -> neck
    3 -> root
    4 -> spine 
    '''

    # root_index = 14

    # new_joints_2d = []

    # # for each frame
    # for joints in joints_2d:
    #     # head top
    #     joints[1] = np.array([(joints[1][i] + joints[2][i]) / 2 for i in range(2)])
    #     # neck
    #     joints[2] = np.array([(joints[5][i] + joints[6][i]) / 2 for i in range(2)])
    #     # root (pelvis)
    #     joints[3] = np.array([(joints[11][i] + joints[12][i]) / 2 for i in range(2)])
    #     # spine
    #     joints[4] = np.array([(joints[2][i] + joints[3][i]) / 2 for i in range(2)])

    #     reorder_idx = [1, 2, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 3, 4, 0]
        
    #     new_joints = np.array([joints[i] for i in reorder_idx])
    #     new_joints_2d.append(new_joints)

    # new_joints_2d = np.array(new_joints_2d)

    #  # set lower body coord = -1
    # lower_body_idx = [9, 10, 12, 13]
    # # for idx in lower_body_idx:
    # #     new_joints_2d[:, idx, :] = -1

    '''
    mmpose convert from 2d coco to 3d mpi_3dhf format

    # pelvis (root) is in the middle of l_hip and r_hip
    keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
    # neck (bottom end of neck) is in the middle of
    # l_shoulder and r_shoulder
    keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
    # spine (centre of torso) is in the middle of neck and root
    keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

    # in COCO, head is in the middle of l_eye and r_eye
    # in PoseTrack18, head is in the middle of head_bottom and head_top
    keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2
    '''

    # keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)

    new_joints_2d = []
    root_index=14

    for joint in joints_2d:
        new_joints = np.zeros((17, 3), dtype=np.float)
        joint = np.array(joint)
        new_joints[14] = (joint[11] + joint[12]) / 2
        new_joints[1] = (joint[5] + joint[6]) / 2
        new_joints[15] = (new_joints[1] + new_joints[14]) / 2
        new_joints[16] = (joint[1] + joint[2]) / 2

        new_joints[0] = (4 * new_joints[16] -
                                        new_joints[1]) / 3
        new_joints[0, 2] = new_joints[16, 2]    

        new_joints[2:14] = joint[[
                    6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15
                ]]                          

        new_joints_2d.append(new_joints)

    new_joints_2d = np.array(new_joints_2d)[:,:,:2]

    joints_3d = np.load(result_3d)[:, :, :3]
    # for idx in lower_body_idx:
    #     joints_3d[:, idx, :] = -1

    centers, scales, joints_2d, joints_3d = get_annotations(joints_2d=new_joints_2d, joints_3d=joints_3d, train_img_size=train_img_size)

    np.savez(
        out_train_file,
        imgname=imgnames,
        center=centers,
        scale=scales,
        part=joints_2d,
        S=joints_3d)

    np.savez(
        out_val_file,
        imgname=imgnames,
        center=centers,
        scale=scales,
        part=joints_2d,
        S=joints_3d)
    

    '''
    create stat pkl file
    '''
    mean_2d, std_2d = get_pose_stats(joints_2d)
    mean_3d, std_3d = get_pose_stats(joints_3d)
    
    # center root
    joints_2d_rel = joints_2d - joints_2d[..., root_index:root_index + 1, :]
    mean_2d_rel, std_2d_rel = get_pose_stats(joints_2d_rel)
    mean_2d_rel[root_index] = mean_2d[root_index]
    std_2d_rel[root_index] = std_2d[root_index]

    joints_3d_rel = joints_3d - joints_3d[..., root_index:root_index + 1, :]
    mean_3d_rel, std_3d_rel = get_pose_stats(joints_3d_rel)
    mean_3d_rel[root_index] = mean_3d[root_index]
    std_3d_rel[root_index] = std_3d[root_index]

    stats = {
        'joint3d_stats': {
            'mean': mean_3d,
            'std': std_3d
        },
        'joint2d_stats': {
            'mean': mean_2d,
            'std': std_2d
        },
        'joint3d_rel_stats': {
            'mean': mean_3d_rel,
            'std': std_3d_rel
        },
        'joint2d_rel_stats': {
            'mean': mean_2d_rel,
            'std': std_2d_rel
        }
    }

    for name, stat_dict in stats.items():
        out_file = osp.join(save_dir, f'annotations/{name}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump(stat_dict, f)
        print(f'Create statistic data file: {out_file}')

    cam_params = dict(
        S1_Seq1_Cam0 = dict(
            w=train_img_size[0],
            h=train_img_size[1],
            name='camera_1'
            )
    )

    with open(osp.join(save_dir, 'annotations/cameras_train.pkl'), 'wb') as f:
        pickle.dump(cam_params, f)
    with open(osp.join(save_dir, 'annotations/cameras_test.pkl'), 'wb') as f:
        pickle.dump(cam_params, f)
        