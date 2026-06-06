# from pathlib import Path
import shutil
from termios import PARENB

def process_and_clean_turnaround(raw_dir, processed_dir):
    """
    Safely migrates ONLY the required train, valid, and test dataset splits
    from raw folder to processed folder, ignoring extraneous metadata files.
    """
    raw_root = raw_dir / "airport-turnaround"
    processed_root = processed_dir / "airport-turnaround"


    # Define explicitly the directory names present in raw workspace
    splits = ["train", "valid", "test"]

    print("🚚 Isolating and migrating clean Airport Turnaround splits...")
    total_splits_moved = 0

    for split in splits:
        src_split_img = raw_root / split / "images"
        src_split_lbl = raw_root / split / "labels"

        target_folder_name = 'val' if split == 'valid' else split

        dest_split_img = processed_root / target_folder_name / "images"
        dest_split_lbl = processed_root / target_folder_name / "labels"

        if src_split_img.exists() and src_split_lbl.exists():
            dest_split_img.mkdir(parents=True, exist_ok=True)
            dest_split_lbl.mkdir(parents=True, exist_ok=True)

            shutil.copytree(src_split_img, dest_split_img, dirs_exist_ok=True)
            shutil.copytree(src_split_lbl, dest_split_lbl, dirs_exist_ok=True)
    print(f"✅ Turnaround Stream completely standardized for YOLO training.\n")



# def copy_files(raw_dir, interim_dir, processed_dir):
#     """
#     TASK:
#     Copy the processed datasets to ~/data/processed folder.
#     """
#     raw_dir = Path(raw_dir)
#     interim_dir = Path(interim_dir)
#     processed_dir = Path(processed_dir)

#     datasets = ["airport-turnaround", "ppe-compliance", "fod-data"]

#     for data in datasets:
#         # Define source path
#         if data != "fod-data":
#             src_dir = raw_dir / data
#         else:
#             src_dir = interim_dir / data
#         # Create destination folder
#         dest_dir = processed_dir / data
#         # Copy folder
#         shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)         
        
#     print("✅ Completed storing the processed datasets.")
    