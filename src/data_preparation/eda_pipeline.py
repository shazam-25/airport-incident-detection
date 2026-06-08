import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize':14})

def analyze_dataset_stream(stream_path, stream_name, class_names, base_path):
    """
    TASK:
    Performs an in-depth spatial, statistical, and structural EDA
    on a specified object detection dataset stream.
    """
    # print(f"\n" + "="*80)
    print(f"📊 STARTING IN-DEPTH EDA FOR {stream_name.upper()}")

    stream_root = Path(stream_path)
    if not stream_root.exists():
        print(f"❌ Error: Stream path does not exists at {stream_root}")
        return None

    splits = ['train', 'val', 'test']
    all_box_records = []
    split_image_counts = {}
    total_objects_per_split = {s: 0 for s in splits}

    # Track complexity (boxes per image) specifically for the training set
    train_boxes_per_image = []

    # 1. Scan across Train, Val, and Test splits for structural counts
    for split in splits:
        img_dir = stream_root / split / "images"
        lbl_dir = stream_root / split / "labels"

        # Count images safely
        if img_dir.exists():
            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
            split_image_counts[split] = len(img_files)
        else:
            split_image_counts[split] = 0
            continue

        # Parse labels if they exist
        if lbl_dir.exists():
            label_files = list(lbl_dir.glob("*.txt"))

            for lbl_file in label_files:
                img_box_count = 0
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_c, y_c, w_box, h_box = map(float, parts[1:])

                            total_objects_per_split[split] += 1
                            img_box_count += 1

                            # Collect deep metrics exclusively from the 'train' split to understand model inputs
                            if split == 'train':
                                all_box_records.append({
                                    'class_id': class_id,
                                    'class_name': class_names[class_id] if class_id < len(class_names) else f"ID_{class_id}",
                                    'x_center': x_c, 'y_center': y_c,
                                    'width': w_box, 'height': h_box,
                                    'aspect_ratio': w_box / (h_box + 1e-6),
                                    'relative_area': w_box * h_box
                                })
                if split == 'train':
                    train_boxes_per_image.append(img_box_count)
    
    # Handle edge case where images exist but have zero annotations (background images)
    # Fill missing values for images that had no matching .txt label file
    if split_image_counts.get('train', 0) > len(train_boxes_per_image):
        background_count = split_image_counts['train'] - len(train_boxes_per_image)
        train_boxes_per_image.extend([0] * background_count)

    # 2. Render Macro Structural Summary Table
    summary_df = pd.DataFrame({
        'Split': ["Train", "Val", "Test"],
        'Image Count': [split_image_counts['train'], split_image_counts['val'], split_image_counts['test']],
        'Total Bounding Boxes': [total_objects_per_split['train'], total_objects_per_split['val'], total_objects_per_split['test']]
    })
    print("\n📈 DATASET SPLIT INFRASTRUCTURE PROFILE:")
    print(summary_df.to_string(index=False))

    if not all_box_records:
        print(f"⚠️ No training label data available to execute advanced spatial EDA for {stream_name}.")
        return summary_df

    df_boxes = pd.DataFrame(all_box_records)

    # Create or check if reports directory exists
    report_dir = Path(base_path/"reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Create or check directory exists to store EDA plots of the streams
    stream_dir = report_dir / stream_name
    stream_dir.mkdir(parents=True, exist_ok=True)

    # 3. VISUALIZATION A: Class Distributions (Train Split)
    filename = "class_distribution.png"
    plt.figure(figsize=(12,5))
    filtered_df_boxes = df_boxes[df_boxes['class_name'].isin(class_names)]
    class_order = pd.Series(filtered_df_boxes['class_name']).value_counts().index
    sns.countplot(data=df_boxes, x='class_name', order=class_order, color='orange', legend=False, edgecolor='black')
    plt.title(f"Class Instance Distribution Profile - {stream_name} (Train Split)", fontweight='bold')
    plt.xlabel("Target Classification Name")
    plt.ylabel("Total Counted Instances")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(stream_dir, filename))
    plt.close()

    # 4. VISUALIZATION B: Scene Complexity Histogram (Objects per Frame)
    # filename = 'histogram.jpg'
    # plt.figure(figsize=(12,2))
    # sns.histplot(train_boxes_per_image, binwidth=1, kde=True, color='teal', edgecolor='black')
    mean_val = float(np.mean(train_boxes_per_image))
    # plt.axvline(mean_val, color='darkorange', linestyle='--', linewidth=2, label=f"Mean Density: {mean_val:.2f} obj/frame")
    # plt.title(f"Scene Complexity Analysis (Clutter Density) - {stream_name}", fontweight='bold')
    # plt.xlabel("Number of Objects Packaged in a Single Frame")
    # plt.ylabel("Frame Frequencies")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join(stream_dir, filename))
    # plt.close()

    # 5. VISUALIZATION C: 2D Spatial Anchoring Heatmap
    filename = "centroid_density_map.png"
    plt.figure(figsize=(6, 5))
    sns.kdeplot(data=df_boxes, x='x_center', y='y_center', fill=True, cmap='rocket', thresh=0.02, levels=12)
    plt.xlim(0, 1)
    plt.ylim(1, 0)  # Invert Y-axis to match pixel space (0,0 is top-left in computer vision)
    plt.title(f"Spatial Anchor Centroid Density Map - {stream_name}", fontweight='bold')
    plt.xlabel("Normalized Horizontal Frame Delta (X)")
    plt.ylabel("Normalized Vertical Frame Delta (Y)")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(stream_dir, filename))
    plt.close()

    # 6. VISUALIZATION D: Object Scale Scatter Plot (Width vs Height)
    filename = "geometric_object_scale_profile.png"
    plt.figure(figsize=(13, 6))
    sns.scatterplot(data=df_boxes, x='width', y='height', hue='class_name', alpha=0.4, s=15, palette='tab10', edgecolor=None)
    plt.title(f"Geometric Object Scale Profile (Aspect Ratio Check) - {stream_name}", fontweight='bold', pad=15)
    plt.xlabel("Normalized Object Width (W)", fontweight='semibold', labelpad=8)
    plt.ylabel("Normalized Object Height (H)", fontweight='semibold', labelpad=8)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper left', title='Classes', title_fontsize='11', fontsize='10', frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(stream_dir, filename))
    plt.close()

    # Log Macro Findings Summary
    print(f"\n🔍 INSIGHT SUMMARY FOR {stream_name.upper()}:")
    print(f"    - Average Scene Congestion Density: {mean_val:.2f} objects per frame.")
    print(f"    - Mean Object Spatial Footprint: {df_boxes['relative_area'].median() * 100:.4f}% of image total frame canvas area.")
    print(f"    - Max Aspect Ratio Skew: {df_boxes['aspect_ratio'].max():.2f} (Extreme wide bounding width).")
    # print("="*80)
    print("="*80 + "\n")

    return summary_df