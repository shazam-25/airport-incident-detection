import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

random.seed(42)

# def extract_class_from_xml(xml_dir):
#     """
#     Helper function to extract class names from XML files.
#     """
#     xml_dir = Path(xml_dir)
#     unique_classes = set()

#     xml_files = list(xml_dir.glob("*.xml"))

#     if not xml_files:
#         print(f"❗ No XML files found in path: {xml_dir}\n\tCheck folder structure.")

#     for xml_file in xml_files:
#         try:
#             tree = ET.parse(xml_file)
#             root = tree.getroot()

#             # Find the class names by parsing the XML file
#             for obj in root.iter("object"):
#                 name_ele = obj.find("name")
#                 if name_ele is None or name_ele.text is None:
#                     continue
#                 class_name = name_ele.text.strip()
#                 unique_classes.add(class_name)

#         except Exception as e:
#             print(f"❗ Exception {xml_file}:\n\t{e}")
#             continue
    
#     class_names = sorted(list(unique_classes))

#     return class_names

def convert_bbox_to_yolo(size, box):
    """
    Standard coordinate normalization.
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x*dw, y*dh, w*dw, h*dh)

def process_and_split_fod(raw_dir, processed_dir):
    """
    Parse Pascal-VOC XML annotations from raw FOD files, converts them to YOLO
    with explicit string spaces, and applies a balanced train/val/test split.
    """
    raw_root = Path(raw_dir) / "fod-data/FODPascalVOCFormat-V.2.1/VOC2007"
    processed_root = Path(processed_dir) / "fod-data"

    xml_dir = raw_root / "Annotations"
    img_dir = raw_root / "JPEGImages"

    if not xml_dir.exists():
        print(f"❌ Error: Cannot track down FOD directory positions at {xml_dir}")
        return []
    
    # Dynamically read unique class instances to compile dictionary indices
    unique_classes = set()
    for xml_file in xml_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.iter("object"):
            name_ele = obj.find("name")
            if name_ele is not None and name_ele.text is not None:
                unique_classes.add(name_ele.text.strip())
    class_names = sorted(list(unique_classes))

    # Define temporary workspaces
    temp_labels = processed_root / "temp_labels"
    temp_labels.mkdir(parents=True, exist_ok=True)

    print("⌛ Translating FOD Pascal-VOC XML arrays into clean YOLO files...")
    valid_stems = []

    for xml_file in xml_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_size = root.find("size")

        if img_size is not None:
            width_el = img_size.find("width")
            height_el = img_size.find("height")
            # Verify both elements and their text attributed exists
            if (width_el is not None and width_el.text is not None) and (height_el is not None and height_el.text is not None):
                w, h = int(width_el.text), int(height_el.text)
            # Skip images with corrupted resolution attributes
            if w == 0 or h == 0: continue

        txt_file_path = temp_labels / f"{xml_file.stem}.txt"
        has_objects = False

        with open(txt_file_path, 'w') as out_file:
            for obj in root.iter("object"):
                # Safely find the name element and its text
                name_ele = obj.find("name")
                if name_ele is None or name_ele.text is None: continue
                cls_name = name_ele.text
                if cls_name not in class_names: continue
                cls_id = class_names.index(cls_name)
                # Safely find the bounding box element
                xml_box = obj.find("bndbox")
                if xml_box is None: continue
                # Extract and verify all four coordinate elements
                xmin_el = xml_box.find("xmin")
                xmax_el = xml_box.find("xmax")
                ymin_el = xml_box.find("ymin")
                ymax_el = xml_box.find("ymax")
                # Ensure none of the coordinates or their text values are missing
                if (
                    xmin_el is not None and xmin_el.text is not None and
                    xmax_el is not None and xmax_el.text is not None and
                    ymin_el is not None and ymin_el.text is not None and
                    ymax_el is not None and ymax_el.text is not None
                ):
                    b = (float(xmin_el.text), float(xmax_el.text),
                         float(ymin_el.text), float(ymax_el.text))
                    
                bb = convert_bbox_to_yolo((w, h), b)
                # Ensure an explicit trailing space separate is written in class ID
                out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + "\n")
                has_objects = True
                # try:
                #     formatted_coords = " ".join([f"{float(a):.6f}" for a in bb])
                #     out_file.write(f"{cls_id} {formatted_coords}\n")
                #     has_objects = True
                # except ValueError as e:
                #     print(f"Skipping export! 'bb' contains non-numeric data: {bb}. Error {e}")
                # # out_file.write(f"{cls_id} " + " ".join([f"{float(a):.6f}" for a in bb]) + "\n")
                
        if has_objects:
            valid_stems.append(xml_file.stem)
        else:
            if txt_file_path.exists():
                txt_file_path.unlink()
    
    # Create Train (80%), Val (10%), Test (10%) splits
    random.shuffle(valid_stems)
    total = len(valid_stems)
    idx_train = int(total * 0.8)
    idx_val = int(total * 0.9)

    splits = {
        'train': valid_stems[:idx_train],
        'val': valid_stems[idx_train:idx_val],
        'test': valid_stems[idx_val:]
    }

    print("✂️ Distributing structured assets into target partitions...")
    for split_name, stem_list in splits.items():
        dest_img = processed_root / split_name / "images"
        dest_lbl = processed_root / split_name / "labels"
        dest_img.mkdir(parents=True, exist_ok=True)
        dest_lbl.mkdir(parents=True, exist_ok=True)

        for stem in stem_list:
            src_txt = temp_labels / f"{stem}.txt"
            if src_txt.exists():
                shutil.move(str(src_txt), str(dest_lbl / src_txt.name))

            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                src_img = img_dir / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy(str(src_img), str(dest_img / src_img.name))
                    break
    shutil.rmtree(temp_labels, ignore_errors=True)
    print(f"✅ Generated FOD split. Total valid records migrated: {total}\n")
    return class_names

# class Format_FOD_Data:

#     """
#     TASK:
#     1. Extract FOD Images & XML Labels.
#     2. Convert XML Labels to YOLO compatible.
#     3. Split FOD data into Train, Val and Test.
#     """

#     def __init__(self):
#         """
#         Initialize source directory & destination directory.
#         """
#         self.source_dir = Path("/teamspace/studios/this_studio/airport-incident-detection/data/raw/fod-data/FODPascalVOCFormat-V.2.1/VOC2007")
#         self.dest_dir = Path("/teamspace/studios/this_studio/airport-incident-detection/data/interim/fod-data")
#         self.dest_dir.mkdir(parents=True, exist_ok=True)
#         print(f"🏗️ Extracting FOD Image & XML files:(📁Files are stored in Intermediate Repository)")
        

#     def extract_fod_images(self):
#         """
#         TASK:
#         1. Extract FOD image files.
#         2. Extract FOD XML Annotation files.
#         """
#         if self.source_dir.exists():
#             print(f"🚚 Copying FOD Image files...")
#             raw_image_dir = self.source_dir/"JPEGImages"
#             interim_image_dir = self.dest_dir/"images"
#             interim_image_dir.mkdir(parents=True, exist_ok=True)

#             if raw_image_dir.exists():
#                 for file in raw_image_dir.iterdir():
#                     if file.is_file() and file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
#                         shutil.copy(str(file), str(interim_image_dir/file.name))
        
#         print("✅ Copied all FOD-A images to Intermediate storage!")
            

#     def xml_to_yolo(self):
#         """
#         TASK:
#         1. Extract class names from XML files.
#         2. Create a txt file for corresponding XMl file.
#         3. Write the bounding box coordinates in txt file.
#         """
#         xml_dir = Path(self.source_dir/"Annotations")
#         txt_path = Path(self.dest_dir/"labels")
#         txt_path.mkdir(parents=True, exist_ok=True)
#         # Extract class names
#         class_names = extract_class_from_xml(xml_dir)

#         print(f"⏳ Converting to FOD-A Labels to YOLO format...")
#         valid_image_stems = []

#         for xml_file in xml_dir.glob("*.xml"):
#             tree = ET.parse(xml_file)
#             root = tree.getroot()
#             img_size = root.find("size")
#             if img_size is not None:
#                 width_ele = img_size.find("width")
#                 height_ele = img_size.find("height")
#                 # Verify both elements and their text attributes exist
#                 if width_ele is not None and width_ele.text is not None:
#                     w = int(width_ele.text)
#                 if height_ele is not None and height_ele.text is not None:
#                     h = int(height_ele.text)
                
#                 # Skip images with corrupted or unreadable resolution attributes
#                 if w == 0 or h == 0:
#                     continue

#             # Create corresponding text file
#             txt_file_path = txt_path / f"{xml_file.stem}.txt"
#             has_valid_objects = False

#             with open(txt_file_path, 'w') as output_file:
#                 for obj in root.iter("object"):
#                     # Safely find the name element and its text
#                     name_ele = obj.find("name")
#                     if name_ele is None or name_ele.text is None:
#                         continue
#                     cls_name = name_ele.text
#                     if cls_name not in class_names:
#                         continue
#                     cls_id = class_names.index(cls_name)
                    
#                     # Safely find the bounding box element
#                     xml_box = obj.find("bndbox")
#                     if xml_box is None:
#                         continue

#                     # Extract and verify all four coordinate elements
#                     xmin_el = xml_box.find("xmin")
#                     xmax_el = xml_box.find("xmax")
#                     ymin_el = xml_box.find("ymin")
#                     ymax_el = xml_box.find("ymax")

#                     # Ensure none of the coordinates or their test values are missing
#                     if (
#                         xmin_el is not None and xmin_el.text is not None and
#                         xmax_el is not None and xmax_el.text is not None and
#                         ymin_el is not None and ymin_el.text is not None and
#                         ymax_el is not None and ymax_el.text is not None
#                     ):
#                         b = (
#                             float(xmin_el.text), float(xmax_el.text),
#                             float(ymin_el.text), float(ymax_el.text)
#                         )
                    
#                     bb = convert_bbox_to_yolo((w,h), b)
#                     output_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + "\n")
#                     has_valid_objects = True
                
#             if has_valid_objects:
#                 valid_image_stems.append(xml_file.stem)
#             else:
#                 if txt_file_path.exists():
#                     txt_file_path.unlink()

#         print(f"✅ Converted FOD XML Labels to YOLO and saved in Intermediate Storage!")


#     def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
#         """
#         TASK:
#         1. Take loose images and converted labels from the Intermediate FOD dataset.
#         2. Split them into balanced train, validation, and test subsets.
#         """
#         print("✂️ Partitioning FOD-A into train/val/test subsets...")

#         interim_image_dir = Path(self.dest_dir/"images")
#         interim_label_dir = Path(self.dest_dir/"labels")

#         # Collect all available images currently inside the interim folder.
#         all_images = sorted(list(interim_image_dir.glob("*.jpg")) +
#                         list(interim_image_dir.glob("*.jpeg")) +
#                         list(interim_image_dir.glob("*.png")))
        
#         if not all_images:
#             print("⚠️ No loose images found to split. (They might already be split!)")
#             return

#         random.shuffle(all_images)

#         total = len(all_images)
#         idx_train = int(total * train_ratio)
#         idx_val = int(total * (train_ratio + val_ratio))

#         splits = {
#             'train': all_images[:idx_train],
#             'valid': all_images[idx_train:idx_val],
#             'test': all_images[idx_val:]
#         }

#         for split_name, image_list in splits.items():
#             dest_img_dir = self.dest_dir / split_name / "images"
#             dest_lbl_dir = self.dest_dir / split_name / "labels"
#             dest_img_dir.mkdir(parents=True, exist_ok=True)
#             dest_lbl_dir.mkdir(parents=True, exist_ok=True)

#             for img_path in image_list:
#                 lbl_path = interim_label_dir / f"{img_path.stem}.txt"

#                 # Copy the image file
#                 shutil.copy(str(img_path), str(dest_img_dir/img_path.name))
#                 # Copy the corresponding tracking label text file
#                 shutil.copy(str(lbl_path), str(dest_lbl_dir/lbl_path.name))

#         # Remove images & labels directory
#         shutil.rmtree(interim_image_dir, ignore_errors=True)
#         shutil.rmtree(interim_label_dir, ignore_errors=True)

#         print(f"✅ FOD Splits generated! Total images: \n{total} -> (Train: {idx_train}, Val: {idx_val - idx_train}, Test: {total - idx_val})")
#         # print("🧹 Removed intermediate fod-a stream data.")

