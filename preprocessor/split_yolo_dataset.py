import os
import shutil
import glob
import random
from pathlib import Path
from tqdm import tqdm
import argparse

def split_by_sequences(source_dir='/workspace/dataset/processed', 
                       output_dir='/workspace/dataset/yolodataset',
                       train_ratio=0.8, seed=42, use_symlink=False,
                       cameras=None):
    
    random.seed(seed)
    
    # Default cameras if not specified
    if cameras is None:
        cameras = ['cam01']
    
    print(f"Processing cameras: {cameras}")
    
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    # Collect and split sequences per dataset (각 0차 폴더별로 8:2 분할)
    train_sequences = []
    val_sequences = []
    
    for dataset in ['03_grappling2', '13_mma2']:
        dataset_path = os.path.join(source_dir, dataset)
        sequences = sorted([d for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d))])
        
        # Shuffle and split within each dataset
        random.shuffle(sequences)
        split_idx = int(len(sequences) * train_ratio)
        
        dataset_train = [(dataset, seq) for seq in sequences[:split_idx]]
        dataset_val = [(dataset, seq) for seq in sequences[split_idx:]]
        
        train_sequences.extend(dataset_train)
        val_sequences.extend(dataset_val)
        
        print(f"{dataset}: {len(sequences)} sequences (train: {len(dataset_train)}, val: {len(dataset_val)})")
    
    print(f"\nTotal sequences: {len(train_sequences) + len(val_sequences)}")
    
    print(f"\nTrain sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    
    print(f"\nMode: {'Symbolic links' if use_symlink else 'Copy files'}")
    
    # Process train sequences sequentially
    print("\nProcessing train sequences...")
    train_count = 0
    for dataset, seq in tqdm(train_sequences, desc="Train"):
        count = copy_sequence_data(source_dir, dataset, seq, output_dir, 'train', use_symlink, cameras)
        train_count += count
    
    # Process val sequences sequentially
    print("Processing val sequences...")
    val_count = 0
    for dataset, seq in tqdm(val_sequences, desc="Val"):
        count = copy_sequence_data(source_dir, dataset, seq, output_dir, 'val', use_symlink, cameras)
        val_count += count
    
    # Create data.yaml
    with open(f"{output_dir}/data.yaml", 'w') as f:
        f.write(f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
names:
  0: player
  1: crowd
nc: 2
""")
    
    print(f"\n{'='*80}")
    print(f"Dataset created!")
    print(f"Train sequences: {len(train_sequences)} ({train_count} files)")
    print(f"Val sequences: {len(val_sequences)} ({val_count} files)")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")


def copy_sequence_data(source_dir, dataset, seq, output_dir, split, use_symlink=False, cameras=None):
    """Copy or symlink all images and labels from one sequence to train or val"""
    
    # Default cameras if not specified
    if cameras is None:
        cameras = ['cam01']
    
    seq_path = os.path.join(source_dir, dataset, seq)
    bbox_dir = os.path.join(seq_path, 'processed_data', 'bbox_player_crowd')
    exo_dir = os.path.join(seq_path, 'exo')
    
    if not os.path.exists(bbox_dir):
        return 0
    
    file_count = 0
    
    # Process only specified cameras
    for cam in cameras:
        label_dir = os.path.join(bbox_dir, cam)
        img_dir = os.path.join(exo_dir, cam, 'images')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            continue
        
        # Process all files
        for label_file in glob.glob(f"{label_dir}/*.txt"):
            frame = os.path.basename(label_file).replace('.txt', '')
            img_file = f"{img_dir}/{frame}.jpg"
            
            if os.path.exists(img_file):
                unique_name = f"{dataset}_{seq}_{cam}_{frame}"
                img_dst = f"{output_dir}/images/{split}/{unique_name}.jpg"
                label_dst = f"{output_dir}/labels/{split}/{unique_name}.txt"
                
                # Use symlink or copy
                if use_symlink:
                    # Create absolute path symlinks
                    os.symlink(os.path.abspath(img_file), img_dst)
                    os.symlink(os.path.abspath(label_file), label_dst)
                else:
                    shutil.copy(img_file, img_dst)
                    shutil.copy(label_file, label_dst)
                
                file_count += 1
    
    return file_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split YOLO dataset by sequences')
    parser.add_argument('--source_dir', default='/workspace/dataset/processed', help='Source directory containing processed data')
    parser.add_argument('--output_dir', default='/workspace/dataset/yolodataset', help='Output directory for YOLO dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_symlink', action='store_true', help='Use symbolic links instead of copying files')
    parser.add_argument('--cameras', nargs='+', default=['cam01'], help='Cameras to process (default: cam01)')
    
    args = parser.parse_args()
    
    split_by_sequences(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        use_symlink=args.use_symlink,
        cameras=args.cameras
    )
