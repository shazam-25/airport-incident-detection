import shutil
from pathlib import Path
from tokenize import RARROW

def process_and_clean_ppe(raw_dir, processed_dir):
    """
    Filters raw PPE annotations to retain exactly 4 classes:
    Safety Vest, Ear Protectors, Without Safety Vest, Without Ear Protectors.
    Maps them to sequential IDs 0, 1, 2, 3 and handles 'valid' name mapping.
    """
    raw_root = Path(raw_dir) / "ppe-compliance"
    processed_root = Path(processed_dir) / "ppe-compliance"

    # 1. Define the explicit 4-class target matrix mapping
    TARGET_MAP = {
        0: 0,   # 'Ear Protectors'          -> becomes 0
        8: 1,   # 'Safety Vest'             -> becomes 1
        9: 2,   # 'Without Ear Protectors'  -> becomes 2
        17: 3   # 'Without Safety Vest'     -> becomes 3
    }

    splits = ["train", "valid", "test"]
    print("🧹 Extracting and re-indexing the 4 target PPE compliance classes...")

    total_files_processed = 0

    for split in splits:
        src_img_dir = raw_root / split / "images"
        src_lbl_dir = raw_root / split / "labels"

        # Standardize 'valid' directory naming convention to YOLO's required 'val'
        target_split_name = 'val' if split == 'valid' else split
        dest_img_dir = processed_root / target_split_name / "images"
        dest_lbl_dir = processed_root / target_split_name / "labels"

        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_lbl_dir.exists():
            continue

        for txt_file in src_lbl_dir.glob("*.txt"):
            valid_lines = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                raw_class_id = int(parts[0])

                # Check if the object belongs to one of the 4 target classes
                if raw_class_id in TARGET_MAP:
                    new_id = TARGET_MAP[raw_class_id]
                    # Format line with a single trailing space after the clean class ID
                    clean_line = f"{new_id} " + " ".join(parts[1:]) + "\n"
                    valid_lines.append(clean_line)

            # Save the file and copy the image only if relevant target classes exist
            if valid_lines:
                with open(dest_lbl_dir / txt_file.name, 'w') as f:
                    f.writelines(valid_lines)
                
                # Copy corresponding image asset safely
                img_extensions = [".jpg", ".jpeg", ".png", ".JPG"]
                for ext in img_extensions:
                    img_name = f"{txt_file.stem}{ext}"
                    if (src_img_dir / img_name).exists():
                        shutil.copy(str(src_img_dir / img_name), str(dest_img_dir / img_name))
                        break
                
                total_files_processed += 1
    print(f"✅ Successfully isolated {total_files_processed} custom filtered files for the PPE Stream!\n")
