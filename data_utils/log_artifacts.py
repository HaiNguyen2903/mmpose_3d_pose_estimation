import wandb

run = wandb.init(project = 'Openmmpose-3d-pose-estimation', tags = ['log video inference'])

run.name = 'log video inference'

# infer_old = wandb.Artifact('detect_v2_reid_v2', type='Inference_videos')

# infer_old.add_dir('/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v2_reid_v2')

# run.log_artifact(infer_old)


infer_new = wandb.Artifact('test', type='3D Pose Inferences')

infer_new.add_file('/home/ducanh/hain/code/mmpose_3d_pose_estimation/vis_result/vis_yoga_15s.mp4')

run.log_artifact(infer_new)