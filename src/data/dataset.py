# Libraries
import os
from typing import override
import cv2
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

class AirportMultiTaskDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=640):
        """
        Multi-Stream Ingestion Pipeline for Airport Safety & Turnaround Moniroting.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.samples = []

        # Explicit Task ID Mapping
        self.tasks = ["turnaround", "ppe", "fod"]

        print(f"---🔍 INITIALIZED MULTI-TASK DATASET FOR SPLIT: [{split.upper()}] ---")
        for task_idx, task_name in enumerate(self.tasks):
            img_path = os.path.join(root_dir, task_name, split, 'images')
            lbl_path = os.path.join(root_dir, task_name, split, 'labels')

            task_count = 0
            if os.path.exists(img_path):
                for img_name in os.listdir(img_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'img_path': os.path.join(img_path, img_name),
                            'lbl_path': os.path.join(lbl_path, os.path.splitext(img_name)[0] + '.txt'),
                            'task_id': task_idx
                        })
                        task_count += 1
            print(f"✅ Bound Task [{task_idx}] ({task_name}): Registered {task_count} frames.")

        print(f"📊 Global Dataset Composition: Total registered multi-task sample frames = {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        task_id = sample['task_id']

        # Read raw image matrix safely
        img = cv2.imread(sample['img_path'])
        if img is None:
            raise FileNotFoundError(f"Failed to ingest matrix for file path: {sample['img_path']}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Scale to input spatial resolution dimensions
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Transpose from HWC structure to CHW Tensor format normalized within [0.0, 1.0]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Load spatial bounding box text parameters
        labels = []
        if os.path.exists(sample['lbl_path']):
            with open(sample['lbl_path'], 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        labels.append(parts)    # Structure mapping: [class_idx, x_center, y_center, width, height]

        if len(labels) == 0:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'image': img_tensor,
            'labels': labels_tensor,
            'task_id': torch.tensor(task_id, dtype=torch.long)
        }

def multi_task_collate_fn(batch):
    """
    Custom collate logic designed to stack images into uniform multi-stream batches
    while combining variable target labels shapes safely.
    """
    images = torch.stack([item['image'] for item in batch])
    task_ids = torch.stack([item['task_id'] for item in batch])

    packed_labels = []
    for img_idx, item in enumerate(batch):
        img_labels = item['labels']
        if img_labels.shape[0] > 0:
            # Inject an absolute image reference dimension into column index 0:
            # [batch_image_pointer, class_idx, x_center, y_center, width, height]
            idx_column = torch.full((img_labels.shape[0], 1), img_idx, dtype=torch.float32)
            packed_labels.append(torch.cat((idx_column, img_labels), dim=1))

    if len(packed_labels) > 0:
        packed_labels = torch.cat(packed_labels, dim=0)
    else:
        packed_labels = torch.zeros((0,6), dtype=torch.float32)
    
    return {
        'images': images,
        'labels': packed_labels,
        'task_ids': task_ids
    }




# class AirportMultiTaskDataset(Dataset):
#     def __init__(self, root_dir, split="train"):
#         """
#         Mutil-Stream Ingestion Pipeline for Airport Safety & Turnaround Monitoring.
#         """
#         self.samples = []

#         # Map tasks to a unique index number
#         self.task_mapping = {"turnaround":0, "ppe":1, "fod":2}

#         # Scan eack task folder for images
#         for task_name, task_id in self.task_mapping.items():
#             img_pattern = os.path.join(root_dir, task_name, split, "images", "*.jpg")
#             img_files = glob.glob(img_pattern)

#             for img_file in img_files:
#                 # Find matching YOLO format label text file
#                 lbl_file = img_file.replace("images", "labels").replace(".jpg", ".txt")
#                 self.samples.append({
#                     "image_path": img_file,
#                     "label_path": lbl_file if os.path.exists(lbl_file) else None,
#                     "task_id": task_id
#                 })
#         print(f"[Dataset] Initialized split '{split}'. Found {len(self.samples)} total samples.")

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         # Read image and format it for PyTorch (640x640, Channels First, Normalized to 0-1)
#         image = cv2.imread(sample["image_path"])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (640, 640))
#         image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

#         # Read standard YOLO labels (class, x, y, w, h)
#         labels = []
#         if sample["label_path"]:
#             with open(sample["label_path"], "r") as f:
#                 for line in f.readlines():
#                     parts = list(map(float, line.strip().split()))
#                     if parts:
#                         labels.append(parts)
        
#         labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
#         task_id_tensor = torch.tensor(sample["task_id"], dtype=torch.long)

#         return image_tensor, labels_tensor, task_id_tensor

# def multi_task_collate_fn(batch):
#     """ Custom collate function to handle variable number of bounding boxes per image """
#     images = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#     task_ids = [item[2] for item in batch]
#     return torch.stack(images, dim=0), labels, torch.stack(task_ids, dim=0)