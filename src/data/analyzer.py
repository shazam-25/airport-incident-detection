# Libraries
import yaml
from pathlib import Path
import pandas as pd
from collections import Counter
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import random

# ==============================
# A. GET CLASS NAMES FROM YAML
# ==============================
def extract_classnames_from_yaml(yaml_path):

    """ Extracts class names from .yaml files """

    yaml_path = Path(yaml_path)
    with open(yaml_path, 'r') as stream:
        try:
            data_config = yaml.safe_load(stream)
            # YOLO configs save class names under the key 'names'
            class_names = data_config.get('names', [])
            # Handle directory style class formats {0:'class A', 1:'class B'}
            if isinstance(class_names, dict):
                class_names = {class_names[i] for i in sorted(class_names.keys())}
            return class_names
        
        except yaml.YAMLError as exc:
            print(f"❌ Error reading YAML file: {exc}")

# ==============================
# B. GET OBJECT CLASS DETAILS
# ==============================
def extract_class_analytics(dir_path, stream):

    """ Returns the unique object class count """

    dir_path = Path(dir_path)
    instance_counter = Counter()
    processed_files = 0

    lbl_dir = dir_path / stream / "labels"

    if not lbl_dir.exists():
        print(f"⚠️ Labels directory missing at: {lbl_dir}")
        return 0, instance_counter

    if stream in ['turnaround', 'ppe']:
        yaml_file = dir_path / stream / "data.yaml"
        src_names = extract_classnames_from_yaml(yaml_file)
        if not src_names:
            return instance_counter

        # Build specific lookup tables based on stream type
        if stream == 'turnaround':
            # Map index directly to name list
            ppe_names = {} 
        elif stream == 'ppe':
            req_class = ["Ear Protectors", "Safety Vest", "Without Ear Protectors", "Without Safety Vest"]
            # Key: Integer ID, Value: String Name
            ppe_names = {src_names.index(item): item for item in req_class if item in src_names}

        # ------------------------------
        # PARSE YOLO TEXT ANNOTATIONS
        # ------------------------------
        for txt_file in lbl_dir.glob("*.txt"):
            processed_files += 1
            with open(txt_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue

                    if stream == 'turnaround':
                        lbl_name = src_names[class_id] if class_id < len(src_names) else f"Unknown-ID({class_id})"
                        instance_counter[lbl_name] += 1
                    elif stream == 'ppe':
                        lbl_name = ppe_names.get(class_id, f"Other-Unwanted-PPE-ID({class_id})")
                        instance_counter[lbl_name] += 1
        return processed_files, instance_counter
    
    elif stream == 'fod':
        # -----------------------------------
        # PARSE PASCAL VOC XML ANNOTATIONS
        # -----------------------------------
        for xml_file in lbl_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                processed_files += 1
                for obj in root.findall('object'):
                    name_element = obj.find('name')
                    if name_element is not None and name_element.text is not None:
                        name_el = name_element.text.strip()
                        instance_counter[name_el] += 1
            except Exception:
                continue    # Skip broken XML targets smoothly
    return processed_files, instance_counter

# ===================================================
# C. CONVERT COUNTER OBJECTS TO PANDAS DATAFRAMES
# ===================================================
def counter_to_df(counter_obj, stream_name):

    """ Converts a Counter collection into a sorted DataFrame """
    
    # Create Pandas Dataframe
    df = pd.DataFrame(list(counter_obj.items()), columns=['Class Name', 'Instance Count'])
    df['Stream'] = stream_name

    # Sort from highest frequency to lowest
    return df.sort_values(by='Instance Count', ascending=False)

# ===============================
# D. VERIFY PROCESSED STREAM IMAGES
# ===============================
def plot_processed_stream(processed_stream_path, class_list=None):
    """
    Plots exactly 3 random processed images from the train split side-by-side 
    in a single horizontal row for clear verification.
    """
    img_dir = Path(processed_stream_path) / "train" / "images"
    lbl_dir = Path(processed_stream_path) / "train" / "labels"
    
    # Grab all non-augmented original images to keep the check clean
    images = [f for f in img_dir.glob("*.jpg") if not f.stem.endswith("_aug")]
    if len(images) < 3:
        # Fall back to any images (including augmented ones) if sample size is tiny
        images = list(img_dir.glob("*.jpg"))
        
    if not images:
        print(f"⚠️ No processed files found in {img_dir}. Check your pipeline path outputs.")
        return
        
    # Sample exactly 3 frames
    samples = random.sample(images, min(3, len(images)))
    
    # Create a 1-row, 3-column canvas area
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if len(samples) < 3:
        # Handle cases where less than 3 total images exist safely
        axes = [axes] if len(samples) == 1 else axes

    for i, img_path in enumerate(samples):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                
                # Turn coordinates back to bounding corners
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                
                # Determine display label string
                if class_list and cls_id < len(class_list):
                    label_text = f"{class_list[cls_id]}"
                else:
                    label_text = f"ID: {cls_id}"
                
                # Draw box outline (Green) and a background text strip for readability
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                
        # Draw on subplot axis canvas
        axes[i].imshow(img)
        axes[i].set_title(f"{img_path.name}", fontsize=10)
        axes[i].axis('off')
        
    plt.suptitle(f"Sanity Check Split Verification: {Path(processed_stream_path).name.upper()} Stream", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()