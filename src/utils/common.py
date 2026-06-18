# Libraries
import os
import yaml
from pathlib import Path

# =========================
# A. DIRECTORY STRUCTURE
# =========================
def directory_tree(root_path, prefix="") -> None:

    """ Function to print the directory structure """

    path = Path(root_path)
    # Filter to only include directories
    folders = [f for f in path.iterdir() if f.is_dir()]
    folders.sort(key=lambda x: x.name.lower())

    for i, folder in enumerate(folders):
        is_last = (i == len(folders) - 1)
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{folder.name}")

        # Recurse into the subfolder
        new_prefix = prefix + ("    " if is_last else "│   ")
        directory_tree(folder, new_prefix)
    return None

# ================================================
# B. IMAGE AND LABEL COUNTS IN STREAM DIRECTORY
# ================================================
def file_counter(dir_path, if_split=True) -> None:

    """ Function to print Image & Label file count in a directory """

    dir_path = Path(dir_path)
    streams=['turnaround','ppe','fod']
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    print("="*50)
    print(f" 🔍 {dir_path.stem.upper()} DIRECTORY FILE COUNT AUDIT ")
    print("="*50)

    # File Count based on Streams
    for stream in streams:
        stream_dir = dir_path / stream
        if not stream_dir.exists():
            print(f"❌ Stream Directory Missing: {stream_dir}")
            continue

        img_count = 0
        yolo_lbl_count = 0
        voc_lbl_count = 0

        # Walk through the directory to catch nested folders
        for root, dirs, files in os.walk(stream_dir):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in image_extensions:
                    img_count += 1
                elif ext == '.txt':
                    yolo_lbl_count += 1
                elif ext == '.xml':
                    voc_lbl_count += 1

        print(f"Stream: {stream.upper()}")
        print(f"    ├── Total Images Located: {img_count}")

        # Display labels found depending on their type context
        if voc_lbl_count > 0:
            print(f"    └── Pascal VOC Labels (.xml): {voc_lbl_count}")
        if yolo_lbl_count > 0:
            print(f"    └── Text Format Labels (.txt): {yolo_lbl_count}")
        if yolo_lbl_count == 0 and voc_lbl_count == 0:
            print(f"    └── ⚠️ WARNING: No matching annotation files detected!")
        
        print("-"*50)
    return None

# ===================================
# C. GET CLASS_NAMES FROM YAML FILE
# ===================================
def get_metadata_from_yaml(yaml_path):
    """
    TASK:
    Reads a standard YOLO data.yaml configuration file and 
    return the list of class names dynamically.
    """
    with open(yaml_path, 'r') as stream:
        try:
            data_config = yaml.safe_load(stream)
            # YOLO configs save class names under the key 'names'
            class_names = data_config.get('names', [])

            # Handle dictionary-style class formats {0: 'class A', 1: 'class B'}
            if isinstance(class_names, dict):
                class_names = [class_names[i] for i in sorted(class_names.keys())]
            return class_names
        except yaml.YAMLError as exc:
            print(f"❌ Error reading YAMl file: {exc}")
            return []

    



