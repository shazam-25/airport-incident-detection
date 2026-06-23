# Libraries
import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter, defaultdict
import yaml
from pathlib import Path

# ============================================
# A. CONVERT PASCAL-VOC TO YOLO LABELS
# ============================================
def parse_voc_xml(xml_path, class_mapping):

    """ Converts a raw Pascal VOC XML coordinate map to normalized YOLO metrics """
    
    bboxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width_el, height_el = size.find('width'), size.find('height')
        if (width_el is not None and width_el.text is not None) and (height_el is not None and height_el.text is not None):
                width, height = float(width_el.text), float(height_el.text)
        
        if width == 0 or height == 0:
            return bboxes, 0, 0

        for obj in root.findall('object'):
            name_el = obj.find('name')
            if name_el is None or name_el.text is None: continue
            class_name = name_el.text
            if class_name not in class_mapping:
                continue
            cls_id = class_mapping[class_name]
            
            bndbox = obj.find('bndbox')
            if bndbox is None: continue
            # Extract and verify all four coordinate elements
            xmin_el = bndbox.find("xmin")
            xmax_el = bndbox.find("xmax")
            ymin_el = bndbox.find("ymin")
            ymax_el = bndbox.find("ymax")
            
            # Ensure none of the coordinates or their text values are missing
            if (
                xmin_el is not None and xmin_el.text is not None and
                xmax_el is not None and xmax_el.text is not None and
                ymin_el is not None and ymin_el.text is not None and
                ymax_el is not None and ymax_el.text is not None
            ):
                xmin = float(xmin_el.text)
                ymin = float(ymin_el.text)
                xmax = float(xmax_el.text)
                ymax = float(ymax_el.text)
            
                # Formula: Center points and delta box dimensions normalized by image canvas limits
                x_center = ((xmin + xmax) / 2.0) / width
                y_center = ((ymin + ymax) / 2.0) / height
                w_box = (xmax - xmin) / width
                h_box = (ymax - ymin) / height
            
                bboxes.append((cls_id, x_center, y_center, w_box, h_box))
        return bboxes, width, height
    except Exception as e:
        print(f"   ⚠️ XML Processing error on {Path(xml_path).name}: {e}")
        return [], 0, 0

# ==============================
# B. IMAGE AUGMENTATION 
# ==============================
def apply_augmentation(image, bboxes):

    """ Applies spatial and pixel modifications to the training set """
    
    aug_img = image.copy()
    aug_bboxes = []
    
    # 1. Horizontal Flip (50% Chance)
    flip = random.random() > 0.5
    if flip:
        aug_img = cv2.flip(aug_img, 1)
    
    # 2. Exposure / Brightness Adjustment
    alpha = random.uniform(0.8, 1.2)    # Contrast scaling
    beta = random.randint(-15, 15)      # Brightness bias
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    for bbox in bboxes:
        cls_id, xc, yc, w, h = bbox
        if flip:
            xc = 1.0 - xc   # Flip center-x point horizontally across normalized frame
        aug_bboxes.append((cls_id, xc, yc, w, h))
    
    return aug_img, aug_bboxes


# =============================
# C. STRATIFIED DATA SPLIT
# =============================
def stratified_split_records(parsed_records, target_ratios=(0.7, 0.15, 0.15)):
    """ Executes iterative stratification across multi-label bounding box distributions
    to ensure perfectly balanced train/val/test class breakdowns """
        # 1. Map out which images contain which classes
    class_to_img_indices = defaultdict(list)
    img_to_classes = {}

    for idx, (_, bboxes) in enumerate(parsed_records):
        classes_in_img = set([box[0] for box in bboxes])
        img_to_classes[idx] = classes_in_img
        for cls in classes_in_img:
            class_to_img_indices[cls].append(idx)

    # 2. Sort classes from rarest to most common
    sorted_classes = sorted(class_to_img_indices.keys(), key=lambda c: len(class_to_img_indices[c]))

    # Initialize empty split buckets
    splits = {'train': [], 'val': [], 'test': []}
    assigned_indices = set()

    # Target capacities based on split ratios
    total_imgs = len(parsed_records)
    target_counts = {
        'train': int(total_imgs * target_ratios[0]),
        'val': int(total_imgs * target_ratios[1]),
        'test': total_imgs - int(total_imgs * target_ratios[0]) - int(total_imgs * target_ratios[1])
    }

    # 3. Iteratively distribute images based on rare class requirements
    for cls in sorted_classes:
        img_indices = class_to_img_indices[cls]
        # Shuffle indices to ensure variety within the class pool
        random.seed(42)
        random.shuffle(img_indices)

        for idx in img_indices:
            if idx in assigned_indices:
                continue

            # Determins which bucket needs data most based on current capacity ratios
            train_ratio = len(splits['train']) / max(1, target_counts['train'])
            val_ratio = len(splits['val']) / max(1, target_counts['val'])
            test_ratio = len(splits['test']) / max(1, target_counts['test'])

            if train_ratio <= val_ratio and train_ratio <= test_ratio:
                splits['train'].append(parsed_records[idx])
            elif val_ratio <= train_ratio and val_ratio <= test_ratio:
                splits['val'].append(parsed_records[idx])
            else:
                splits['test'].append(parsed_records[idx])
            
            assigned_indices.add(idx)
    
    # Clean up any unassigned items smoothly
    for idx, record in enumerate(parsed_records):
        if idx not in assigned_indices:
            splits['train'].append(record)
    
    return splits

# =====================================
# ---STREAM PREPROCESSING PIPELINE---
# =====================================
def execute_processing_pipeline(stream_name, raw_root, proc_root, fod_classes=None):
    print("\n" + "="*75)
    print(f"🚩 PROCESSING STREAM IMAGES: {stream_name.upper()}\n" + "="*75)
    raw_dir = Path(raw_root) / stream_name
    proc_dir = Path(proc_root) / stream_name

    # 1. Maps raw PPE text IDs directly to consecutive indices
    ppe_map = {0: 0, 8: 1, 9: 2, 17: 3}
    
    # 2. Maps raw FOD XML text strings directly to consecutive indices
    fod_map = {name: idx for idx, name in enumerate(fod_classes)} if fod_classes else {}
    
    if stream_name == 'fod':
        print(f"🎯 Aligned FOD Mapping Matrix: {fod_map}")

    raw_img_dir = raw_dir / "images"
    raw_lbl_dir = raw_dir / "labels"

    if not raw_img_dir.exists():
        print(f"❌ Images directory not found: {raw_img_dir}")
        return

    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = [f for f in raw_img_dir.iterdir() if f.suffix in valid_extensions]

    parsed_records = []
    native_widths, native_heights, relative_box_areas = [], [], []

    # ---------------------------------------------------
    # PHASE 1: PARSING AND LABEL FILTERING
    # ---------------------------------------------------
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_native, w_native, _ = img.shape
        native_widths.append(w_native)
        native_heights.append(h_native)

        bboxes = []
        
        # 🍏 Case A: FOD (XML strings -> Integers)
        if stream_name == 'fod':
            xml_path = raw_lbl_dir / f"{img_path.stem}.xml"
            if xml_path.exists():
                bboxes, _, _ = parse_voc_xml(xml_path, fod_map)
                
        # 🍏 Case B: Text Annotation Formats (Turnaround & PPE)
        else:
            txt_path = raw_lbl_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) != 5: continue
                        raw_cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])

                        # Turnaround: Pass raw class IDs through completely untouched
                        if stream_name == 'turnaround':
                            bboxes.append((raw_cls, xc, yc, w, h))
                            
                        # PPE: Remap only the 4 target class IDs
                        elif stream_name == 'ppe':
                            if raw_cls in ppe_map:
                                bboxes.append((ppe_map[raw_cls], xc, yc, w, h))
            
        if not bboxes:
            continue  # Skip frames without target labels

        for box in bboxes:
            relative_box_areas.append(box[3] * box[4])
            
        parsed_records.append((img_path, bboxes))
    
    # -----------------------------------------------
    # PHASE 2: DISPLAY GEOMETRIC EXPLORATORY METRICS
    # -----------------------------------------------
    if parsed_records:
        print(f"--- 📊 {stream_name.upper()} STREAM GEOMETRIC ANALYSIS REPORT (RAW) ---")
        print(f"    Total Valid Image Assets    : {len(parsed_records)}")
        print(f"    Mean Image Width            : {np.mean(native_widths):.1f} px (Min: {np.min(native_widths)}, Max: {np.max(native_widths)})")
        print(f"    Mean Image Height           : {np.mean(native_heights):.1f} px (Min: {np.min(native_heights)}, Max: {np.max(native_heights)})")
        print(f"    Bounding Box Area Profile   : Mean: {np.mean(relative_box_areas)*100:.2f}% | Max Object Size: {np.max(relative_box_areas)*100:.2f}% of canvas")
        print("-"*60)
    # -----------------------------------------------------------------
    # PHASE 3: STRATIFIED SPLITTING & EXPORTING
    # -----------------------------------------------------------------
    print("--- ⌛ IMAGE AUGMENTATION & STRATIFICATION SPLIT ---")
    splits = stratified_split_records(parsed_records)

    for split_key, records in splits.items():
        out_img_dir = proc_dir / split_key / "images"
        out_lbl_dir = proc_dir / split_key / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        # split_class_counter = Counter()
        for idx, (img_path, bboxes) in enumerate(records):
            img = cv2.imread(str(img_path))
            resized_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

            orig_stem = img_path.stem
            cv2.imwrite(str(out_img_dir / f"{orig_stem}.jpg"), resized_img)

            with open(out_lbl_dir / f"{orig_stem}.txt", 'w') as f:
                for box in bboxes:
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                    # split_class_counter[box[0]] += 1
            
            if split_key == 'train':
                aug_img, aug_boxes = apply_augmentation(resized_img, bboxes)
                cv2.imwrite(str(out_img_dir / f"{orig_stem}_aug.jpg"), aug_img)
                with open(out_lbl_dir / f"{orig_stem}_aug.txt", 'w') as f:
                    for box in aug_boxes:
                        f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        
        if split_key == 'train':
            print(f"    🔶 [{split_key.upper()}] split augmentation complete.")             
        
        print(f"    📂 [{split_key.upper()}] split complete. Saved {len(records)} [640x640] resized images.")
        # \n      Class IDs distribution: {dict(split_class_counter)}")
        # print("-"*60)
        

# ==================================
# D. YAML CONFIG FILE GENERATOR
# ==================================
def generate_yaml_profiles(proc_dir, stream_name, class_names):
    """ Generates unified config file formats with accurate structural path mappings """
    proc_path = Path(proc_dir)
    config_dir = proc_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        f"{stream_name}.yaml": {
            "path": os.path.join(proc_path, stream_name),
            "train": "train/images", "val": "val/images", "test": "test/images",
            "nc": len(class_names),
            "names": list(class_names)
        }
    }

    for filename, structure in configs.items():
        yaml_path = config_dir / filename
        with open(yaml_path, 'w') as f:
            yaml.dump(structure, f, default_flow_style=False)
        stream_name = filename.replace('.yaml', '')
        print(f"✅ [{stream_name.upper()}] Configuration map : {filename}")

