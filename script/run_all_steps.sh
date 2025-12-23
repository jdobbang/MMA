#!/bin/bash

# Run MMA pipeline
python script/run_mma_pipeline.py --detections detection/test.csv --images-dir dataset/yolodataset/images/val

# Transform coordinates in batch
python script/transform_coordinates.py --batch

# Visualize frames with tracking results
python script/visualize_tracking.py --batch --frames-dir dataset/yolodataset/images/val

# Visualize data in batch
python script/visualize_data.py --batch

# Concatenate tracking results to mp4
python script/concat_to_mp4.py --tracking-dir tracking_results --fps 30
