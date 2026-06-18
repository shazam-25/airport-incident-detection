# Libraries
import os
import shutil
from zipfile import ZipFile
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

# ==========================
# A. DATA EXTRACTION
# ==========================
def fetch_stream_dataset(raw_dir):
    """ Extract & Unzip Stream datasets """

    # ------------------------
    # 1. MAP STREAM TO REPO
    # ------------------------
    stream_manifest = {
        "turnaround": "shazam0k/airport-turnaround-dataset",
        "ppe": "shazam0k/airport-ppe-dataset",
        "fod": "kilogrand/foreign-object-debris-in-airports-fod-a-dataset"
    }
    print("="*60 + "\n▶️ Starting Stream Data Extraction...\n" + "="*60)
    # ------------------------
    # 2. DOWNLOAD & UNZIP
    # ------------------------
    for stream_name, repository_slug in stream_manifest.items():
        destination_path = raw_dir/stream_name
        destination_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📡 Downloading raw data for {stream_name}...")
        os.system(f"kaggle datasets download -d {repository_slug} --path {destination_path}")

        zip_archives = list(destination_path.glob("*.zip"))
        if zip_archives:
            print(f"📦 Unpacking compressed data into {stream_name}")
            with ZipFile(zip_archives[0], 'r') as archive_ref:
                archive_ref.extractall(destination_path)

            # Instantly delete zip file for clean workspace
            zip_archives[0].unlink()
            print(f"🧹 Deleted the zip file of {stream_name} for clean workspace!")

    print(f"✅ Successfully extracted Stream Datasets!\n")

# ===================================
# RESTRUCTURE RAW STREAMS DIRECTORY
# ===================================
def reconstruct_and_clean_raw(raw_dir):

    """ Restructures the raw directory of all streams """
    
    raw_dir = Path(raw_dir)
    print("="*70 + "\n▶️ Starting Raw Dataset Consolidation, Reconstruction & Purge Loop...\n" + "="*70 + "\n")

    # ---------------------------------
    # 1. PROCESS TURNAROUND & PPE 
    # ---------------------------------
    for stream in ['turnaround', 'ppe']:
        stream_dir = raw_dir / stream
        if not stream_dir.exists():
            print(f"⚠️ Directory missing, skipping: {stream_dir}")
            continue

        print(f"📦 Consolidating Multi-Split Streams -> [{stream.upper()}]")

        all_found_images = []
        all_found_labels = []
        garbage_to_delete = []

        # Traverse the nested layput to extract pairs
        for root, dirs, files in os.walk(stream_dir):
            root_path = Path(root)

            if root_path.name in ['train', 'valid', 'test', 'images', 'labels']:
                if root_path not in garbage_to_delete:
                    garbage_to_delete.append(root_path)
                
            for file in files:
                file_path = root_path / file
                ext = file_path.suffix.lower()

                # Check for redundant setup parameters and metadata
                if file in ['README.dataset.txt', 'README.roboflow.txt'] or ext == '.html':
                    if file_path not in garbage_to_delete:
                        garbage_to_delete.append(file_path)
                    continue

                if ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    all_found_images.append(file_path)
                elif ext == '.txt':
                    all_found_labels.append(file_path)
        
        # Standardize target directory structure
        target_img_dir = stream_dir / "images"
        target_lbl_dir = stream_dir / "labels"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Consolidate raw data shards into a flat directory layer
        moved_imgs, moved_lbls = 0, 0
        for img_file in all_found_images:
            if img_file.parent != target_img_dir:
                shutil.move(str(img_file), str(target_img_dir / img_file.name))
                moved_imgs += 1
        for lbl_file in all_found_labels:
            if lbl_file.parent != target_lbl_dir:
                shutil.move(str(lbl_file), str(target_lbl_dir / lbl_file.name))
                moved_lbls += 1
        print(f"    ├── Migrated {moved_imgs} images into unified flat folder layout.")
        print(f"    └── Migrated {moved_lbls} text label annotations into unified flat folder layout.")

        # Purge empty directories and unwanted configuration clutter files
        for trash in sorted(garbage_to_delete, key=lambda p: len(str(p)), reverse=True):
            if trash.exists():
                if trash.is_file():
                    trash.unlink()
                elif trash.is_dir() and not os.listdir(trash):
                    trash.rmdir()

    # ----------------------------
    # 2. PROCESS FOD
    # ----------------------------
    fod_root = raw_dir / "fod"
    if fod_root.exists():
        print("📦 Consolidating Nested Structured Layouts -> [FOD]")

        target_img_dir = fod_root / "images"
        target_lbl_dir = fod_root / "labels"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        all_images = list(fod_root.glob("**/JPEGImages/*.*")) + list(fod_root.glob("**/*.jpg"))
        all_xmls = list(fod_root.glob("**/Annotations/*.xml")) + list(fod_root.glob("**/*.xml"))

        moved_imgs, moved_xmls = 0, 0
        for img in all_images:
            if img.exists() and img.parent != target_img_dir:
                shutil.move(str(img), str(target_img_dir / img.name))
                moved_imgs += 1
        for xml in all_xmls:
            if xml.exists() and xml.parent != target_lbl_dir:
                shutil.move(str(xml), str(target_lbl_dir / xml.name))
                moved_xmls += 1
        print(f"    ├── Flattened {moved_imgs} source images into fod/images/")
        print(f"    └── Flattened {moved_xmls} Pascal VOC XML frames into fod/labels/")

        # Clear out empty residual subdirectories
        for item in fod_root.iterdir():
            if item.is_dir() and item.name not in ['images', 'labels']:
                shutil.rmtree(item)

    print("✅ Raw directory reconstruction and garbage collection purge complete.")

# ==========================
# ---EXTRACTION PIPELINE---
# ==========================
def execute_extraction_pipeline(raw_dir):
    """ Executes the entire extraction & cleaning workflow sequentially """
    raw_dir = Path(raw_dir)
    # 1. Download & extract
    fetch_stream_dataset(raw_dir)
    # 2. Clean & restructure
    reconstruct_and_clean_raw(raw_dir)

# if __name__ == "__main__":
#     raw_path = Path("/teamspace/studios/this_studio/airport-incident-detection/data/raw")
#     execute_extraction_pipeline(raw_path)
