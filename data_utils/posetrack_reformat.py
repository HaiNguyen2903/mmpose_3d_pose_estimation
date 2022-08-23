import os 
import os.path as osp
import json

def isSmaller(box1, box2):
    area1 = (float(box1[2]) - float(box1[0]))*(float(box1[3]) - float(box1[1]))
    area2 = (float(box2[2]) - float(box2[0]))*(float(box2[3]) - float(box2[1]))

    assert area1 > 0
    assert area2 > 0
    return area1 < area2

def xyxy2xywh(bbox):
    x1, y1, x2, y2, conf = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
    return [x1, y1, x2-x1, y2-y1, conf]

if __name__ == '__main__':
    json_path = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/175_30s_2d_anno.json'
    dst_root = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/posetrack18_format'
    detection_save_path = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/posetrack18_format/2dtrack_val_human_detections.json'
    poseval_save_path = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/posetrack18_format/2dtrack_val.json'
    posetrain_save_path = '/home/ducanh/hain/code/mmpose_3d_pose_estimation/2d_results/posetrack18_format/2dtrack_train.json'

    f = open(json_path)
    data = json.load(f)

    detection_content = []
    pose_train_content = []
    pose_val_content = []

    '''
    Generate human detection json file
    '''
    # for i in range(len(data)):
    #     # each frame has 1 person
    #     obj = data[i][0]
    #     bbox = obj['bbox']
    #     bbox = xyxy2xywh(bbox)
        
    #     obj_info = dict(
    #         image_id = "{:05}".format(i),
    #         bbox = bbox[:3],
    #         score = bbox[4],
    #         category_id = 1
    #     )
        
    #     detection_content.append(obj_info)

    # with open(detection_save_path, 'w') as f:
    #     json.dump(detection_content, f)


    
    '''
    Generate pose val json file
    '''
    cate_info = [
        {
        'supercategory': 'person',
        'id': 1,
        'name': 'person',
        'keypoints': 
            ['nose',
            'head_bottom',
            'head_top',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle'],
        'skeleton':
            [[16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7]]
        }
    ]

    imgs_info = []
    annos_info = []

    for i in range(len(data)):
        # create img info for each frame
        img_info = dict(
            has_no_densepose = True,
            is_labeled = False,
            file_name = 'images/val/{:05}.jpg'.format(i),
            nframes = 900,
            frame_id = i,
            id = 1,
            width = 608,
            height = 1080
        )
        imgs_info.append(img_info)


        # create anno info for each frame
        # each frame has 1 person
        obj = data[i][0]
        origin_kpts = obj['keypoints']
        new_kpts = []
        for joint in origin_kpts:
            new_kpts.extend(joint[:2] + [1])

        anno_info = dict(
            keypoints = new_kpts,
            scores = [],
            category_id = 1,
            id = 1,
            iscrowd = False,
            num_keypoints = 17
        )

        annos_info.append(anno_info)

    pose_val_content = dict(
        categories = cate_info,
        images = imgs_info,
        annotations = annos_info
    )

    with open(poseval_save_path, 'w') as f:
        json.dump(pose_val_content, f)

    '''
    Generate pose train json file (similar as pose val json file)
    '''
    cate_info = [
        {
        'supercategory': 'person',
        'id': 1,
        'name': 'person',
        'keypoints': 
            ['nose',
            'head_bottom',
            'head_top',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle'],
        'skeleton':
            [[16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7]]
        }
    ]

    imgs_info = []
    annos_info = []

    for i in range(len(data)):
        # create img info for each frame
        img_info = dict(
            has_no_densepose = True,
            is_labeled = False,
            file_name = 'images/train/{:05}.jpg'.format(i),
            nframes = 900,
            frame_id = i,
            id = 1,
            width = 608,
            height = 1080
        )
        imgs_info.append(img_info)


        # create anno info for each frame
        # each frame has 1 person
        obj = data[i][0]
        origin_kpts = obj['keypoints']
        new_kpts = []
        for joint in origin_kpts:
            new_kpts.extend(joint[:2] + [1])

        anno_info = dict(
            keypoints = new_kpts,
            scores = [],
            category_id = 1,
            id = 1,
            iscrowd = False,
            num_keypoints = 17
        )

        annos_info.append(anno_info)

    pose_val_content = dict(
        categories = cate_info,
        images = imgs_info,
        annotations = annos_info
    )

    with open(posetrain_save_path, 'w') as f:
        json.dump(pose_train_content, f)


    
        