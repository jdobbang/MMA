from ultralytics import YOLO

model = YOLO('yolo11x-pose.pt')

model.train(
    data='/workspace/MMA/dataset/yolo_pose_cropped_dataset/data.yaml',
    epochs=100,
    project='/workspace/MMA/pose_log', # 상위 폴더
    name='mma_overlap_v1',                    # 세부 폴더 명
    exist_ok=True                             # 동일 이름 폴더가 있을 때 덮어쓰기 여부
)