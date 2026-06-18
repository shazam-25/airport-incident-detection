import torch
import torch.nn as nn
from ultralytics import YOLO

class MultiHeadAirportYOLO(nn.Module):
    def __init__(self, num_turnaround_cls=13, num_ppe_cls=4):
        super(MultiHeadAirportYOLO, self).__init__()
        
        # 1. Load Pretrained YOLOv8 Small as our Feature Extraction Backbone
        # We extract features from the SPPF (Spatial Pyramid Pooling Fast) layer or Neck
        yolo_base = YOLO('yolov8s.pt').model
        
        # Extract layers up to the neck (index 22 is typical for YOLOv8 neck output)
        # self.shared_backbone = nn.Sequential(*list(yolo_base.children())[:11])
        self.shared_backbone = nn.ModuleList(list(yolo_base.model[:10])) 
        
        # Freeze early backbone layers to speed up training & stabilize gradients
        for param in self.shared_backbone[:10].parameters():
            param.requires_grad = True  # Keep backbone trainable but stable
            
        # 2. Define Custom Task-Specific Prediction Heads
        # Head 1: Turnaround Object Detection (Returns standard bounding box regression + classification)
        self.turnaround_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_turnaround_cls + 4, kernel_size=1) # class scores + bounding box (x,y,w,h)
        )
        
        # Head 2: PPE Detection (Person, Vest, Ear Protector classification mapped to boxes)
        self.ppe_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_ppe_cls + 4, kernel_size=1)
        )
        
        # Head 3: FOD High-Resolution Anomaly Head 
        # Focuses strictly on a binary target output (0: Clean, 1: Debris Present) or micro-bbox anchor
        self.fod_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1) # Binary classification logit for patch anomaly
        )

    def forward(self, x):
        # FIX 2: Manually save intermediate layer tensors to handle YOLO skip connections
        saved_features = []
        
        # Route through the extracted backbone layers one by one safely
        for i, layer in enumerate(self.shared_backbone):
            # If your custom YOLO configuration needs specific layer lookbacks, 
            # save them into the list here. For indices 0-9, they flow cleanly:
            x = layer(x)
            saved_features.append(x)
            
        # At this point, index 9 output represents your shared backbone features
        shared_features = x 

        # Route identical feature maps to all 3 custom airport task heads simultaneously
        out_turnaround = self.turnaround_head(shared_features)
        out_ppe = self.ppe_head(shared_features)
        out_fod = self.fod_head(shared_features)

        return out_turnaround, out_ppe, out_fod

class JointMultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.5, gamma=2.0):
        super(JointMultiTaskLoss, self).__init__()
        self.alpha = alpha   # Turnaround weight
        self.beta = beta     # PPE weight (boosted to penalize safety drops)
        self.gamma = gamma   # FOD weight (highest weight because anomalies are small/sparse)
        
        # Component loss functions
        self.bbox_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.fod_binary_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        pred_t, pred_p, pred_f = preds
        tgt_t, tgt_p, tgt_f = targets

        # Calculate isolated losses (simplified representation for demonstration)
        loss_turnaround = self.bbox_loss(pred_t[:, :4], tgt_t[:, :4]) + self.cls_loss(pred_t[:, 4:], tgt_t[:, 4])
        loss_ppe = self.bbox_loss(pred_p[:, :4], tgt_p[:, :4]) + self.cls_loss(pred_p[:, 4:], tgt_p[:, 4])
        loss_fod = self.fod_binary_loss(pred_f, tgt_f)

        # Unified backpropagation gradient vector
        total_loss = (self.alpha * loss_turnaround) + (self.beta * loss_ppe) + (self.gamma * loss_fod)
        return total_loss