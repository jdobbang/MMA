import os
import shutil
from pathlib import Path

def cleanup_folders(base_path='/workspace/dataset'):
    """
    1차 하위: 03_grappling2, 13_mma2, 15_mma4 등
    2차 하위: 017_grappling2, 023_mma4 등 (여기서 colmap, ego 삭제)
    3차 하위: processed_data (여기서 pose3d, smpl 삭제)
    """
    
    deleted_folders = []
    errors = []
    
    # 1차 하위 폴더들 (타겟: 03_grappling2, 13_mma2, 15_mma4)
    target_folders = ['03_grappling2', '13_mma2', '15_mma4']
    
    for folder_name in target_folders:
        first_level = Path(base_path) / folder_name
        if not first_level.exists() or not first_level.is_dir():
            print(f"\n[1차] Skipping (not found): {folder_name}")
            continue
            
        print(f"\n[1차] Processing: {first_level.name}")
        
        # 2차 하위 폴더들 찾기
        for second_level in first_level.iterdir():
            if not second_level.is_dir():
                continue
                
            print(f"  [2차] Processing: {second_level.name}")
            
            # 2차 하위에서 colmap, ego 삭제
            for folder_name in ['colmap', 'ego']:
                folder_path = second_level / folder_name
                if folder_path.exists():
                    try:
                        print(f"    Deleting: {folder_path}")
                        shutil.rmtree(folder_path)
                        deleted_folders.append(str(folder_path))
                    except Exception as e:
                        print(f"    Error deleting {folder_path}: {e}")
                        errors.append((str(folder_path), str(e)))
            
            # 3차 하위에서 processed_data 내의 pose3d, smpl 삭제
            processed_data = second_level / 'processed_data'
            if processed_data.exists() and processed_data.is_dir():
                print(f"    [3차] Processing: processed_data")
                for folder_name in ['poses3d', 'smpl']:
                    folder_path = processed_data / folder_name
                    if folder_path.exists():
                        try:
                            print(f"      Deleting: {folder_path}")
                            shutil.rmtree(folder_path)
                            deleted_folders.append(str(folder_path))
                        except Exception as e:
                            print(f"      Error deleting {folder_path}: {e}")
                            errors.append((str(folder_path), str(e)))
    
    print("\n" + "="*60)
    print(f"SUMMARY: Deleted {len(deleted_folders)} folders")
    if errors:
        print(f"Errors: {len(errors)}")
        for path, error in errors:
            print(f"  - {path}: {error}")
    print("="*60)

if __name__ == "__main__":
    cleanup_folders()
