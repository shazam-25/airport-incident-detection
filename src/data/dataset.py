# Libraries
import os
import cv2
import glob
import torch
from torch.utils.data import Dataset, DataLoader

class AirportMultiTaskDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.samples = []

        # Map tasks to a unique index number
        self.task_mapping = {"turnaround":0, "ppe":1, "fod":2}

        # Scan eack task folder for images
        for task_name, task_id in self.task_mapping.items():
            img_pattern = os.path.join(root_dir, task_name, split, "images", "*.jpg")
            img_files = glob.glob(img_pattern)

            for img_file in img_files:
                # Find matching YOLO format label text file
                lbl_file = img_file.replace("images", "labels").replace(".jpg", ".txt")
                self.samples.append({
                    "image_path": img_file,
                    "label_path": lbl_file if os.path.exists(lbl_file) else None,
                    "task_id": task_id
                })
        print(f"[Dataset] Initialized split '{split}'. Found {len(self.samples)} total samples.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Read image and format it for PyTorch (640x640, Channels First, Normalized to 0-1)
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Read standard YOLO labels (class, x, y, w, h)
        labels = []
        if sample["label_path"]:
            with open(sample["label_path"], "r") as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if parts:
                        labels.append(parts)
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        task_id_tensor = torch.tensor(sample["task_id"], dtype=torch.long)

        return image_tensor, labels_tensor, task_id_tensor

def multi_task_collate_fn(batch):
    """ Custom collate function to handle variable number of bounding boxes per image """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    task_ids = [item[2] for item in batch]
    return torch.stack(images, dim=0), labels, torch.stack(task_ids, dim=0)