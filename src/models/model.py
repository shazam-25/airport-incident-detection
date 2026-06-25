import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskAirportNet(nn.Module):
    def __init__(self, num_turnaround=13, num_ppe=4, num_fod=31):
        super().__init__()
        print("--- 🏗️ SYNTHESIZING MULTI-TASK NETWORK ARCHITECTURE ---")

        # 1. Feature Extraction Backbone (Mimic YOLO stype paths)
        # Extract high-resolution features and freeze early layers to save compute
        base_model = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,  # Low-level spatial features
            base_model.layer2   # Mid-level semantic features (128 channels, 80x80 for 640x640 input)
        )

        # Freeze early layer parameters to lock base feature extractors
        for param in self.backbone[:5].parameters():
            param.requires_grad = False
        print(" Locked early backbone layers successfully.")

        # 2. Head 1: Turnaround Detection & Tracking Anchors (13 classes)
        self.turnaround_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 4 + num_turnaround, kernel_size=1)   # [4 box coords + classes]
        )

        # 3. Head 2: PPE Compliance Multi-Class Head (4 classes)
        self.ppe_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 4 + num_ppe, kernel_size=1)
        )

        # 4. Head 3: FOD High-Resolution Tiny-Object Head (31 classes)
        self.fod_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Increased channel capacity for fine features
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 4 + num_fod, kernel_size=1)
        )

        print(" All 3 specialized spatial task branches successfully configured.")

    def forward(self, x, task_id=None):
        # Shared feature map generation
        features = self.backbone(x) # Out shape: [Batch, 128, 80, 80]

        if task_id is not None:
            if task_id == 0: return {"turnaround": self.turnaround_head(features)}
            elif task_id == 1: return {"ppe": self.ppe_head(features)}
            elif task_id == 2: return {"fod": self.fod_head(features)}

        return {
            "turnaround": self.turnaround_head(features),
            "ppe": self.ppe_head(features),
            "fod": self.fod_head(features)
        }


# class ConvBlock(nn.Module):
#     """Standard Conv-BatchNorm-SiLU block used to construct the network layers."""
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

# class SharedBackbone(nn.Module):
#     """
#     Unified Feature Extraction Backbone (~11.5M parameters baseline structure).
#     Downsamples spatial geometry while building rich semantic maps for all 3 tasks.
#     """
#     def __init__(self):
#         super().__init__()
#         # Input: 3 x 640 x 640
#         self.stem = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)        # -> 32 x 320 x 320
#         self.layer1 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)     # -> 64 x 160 x 160
#         self.layer2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)    # -> 128 x 80 x 80
#         self.layer3 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)   # -> 256 x 40 x 40

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x    # This shared latent map is passed to all 3 task heads

# class ObjectDetectionHead(nn.Module):
#     """
#     Specialized task branch for parsing localized object classes and spatial coordinates.
#     """
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.feature_refiner = ConvBlock(in_channels, 128, kernel_size=3, stride=1, padding=1)

#         # 1. Bounding Box Regression Head (Outputs 4 points: x, y, w, h)
#         self.bbox_reg = nn.Conv2d(128, 4, kernel_size=1)

#         # 2. Classification Head (Outputs prediction confidence scores per class)
#         self.cls_score = nn.Conv2d(128, num_classes, kernel_size=1)

#     def forward(self, x):
#         feat = self.feature_refiner(x)
#         reg_out = self.bbox_reg(feat)
#         cls_out = self.cls_score(feat)
#         return {"reg": reg_out, "cls": cls_out}

# class MutliTaskAiportNet(nn.Module):
#     """
#     Unified Multi-Task Learning Network for Airport Airside Monitoring.
#     Integrates 1 shared backbone with 3 specialized prediction heads.
#     """
#     def __init__(self):
#         super().__init__()
#         self.backbone = SharedBackbone()

#         # Head Assignments:
#         # Stream 0: Turnaround (13 classes)
#         self.turnaround_head = ObjectDetectionHead(in_channels=256, num_classes=13)
#         # Stream 1: PPE Compliance (4 classes)
#         self.ppe_head = ObjectDetectionHead(in_channels=256, num_classes=4)
#         # Stream 2: Foreign Object Debris (31 classes)
#         self.fod_head = ObjectDetectionHead(in_channels=256, num_classes=31)

#     def forward(self, x, task_id=None):
#         # 1. Extract shared feature maps
#         features = self.backbone(x)

#         # 2. Dynamic Routing Option based on Task ID (Inference Mode)
#         if task_id is not None:
#             if task_id == 0: return {"turnaround": self.turnaround_head(features)}
#             elif task_id == 1: return {"ppe": self.ppe_head(features)}
#             elif task_id == 2: return {"fod": self.fod_head(features)}
        
#         # 3. Parallel Global Pass Option (Batch Training Mode)
#         return {
#             "turnaround": self.turnaround_head(features),
#             "ppe": self.ppe_head(features),
#             "fod": self.fod_head(features)
#         }


# class MultiHeadAirportYOLO(nn.Module):
#     def __init__(self, num_turnaround_cls=13, num_ppe_cls=4):
#         super(MultiHeadAirportYOLO, self).__init__()
        
#         # 1. Load Pretrained YOLOv8 Small as our Feature Extraction Backbone
#         # We extract features from the SPPF (Spatial Pyramid Pooling Fast) layer or Neck
#         yolo_base = YOLO('yolov8s.pt').model
        
#         # Extract layers up to the neck (index 22 is typical for YOLOv8 neck output)
#         # self.shared_backbone = nn.Sequential(*list(yolo_base.children())[:11])
#         self.shared_backbone = nn.ModuleList(list(yolo_base.model[:10])) 
        
#         # Freeze early backbone layers to speed up training & stabilize gradients
#         for param in self.shared_backbone[:10].parameters():
#             param.requires_grad = True  # Keep backbone trainable but stable
            
#         # 2. Define Custom Task-Specific Prediction Heads
#         # Head 1: Turnaround Object Detection (Returns standard bounding box regression + classification)
#         self.turnaround_head = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, num_turnaround_cls + 4, kernel_size=1) # class scores + bounding box (x,y,w,h)
#         )
        
#         # Head 2: PPE Detection (Person, Vest, Ear Protector classification mapped to boxes)
#         self.ppe_head = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, num_ppe_cls + 4, kernel_size=1)
#         )
        
#         # Head 3: FOD High-Resolution Anomaly Head 
#         # Focuses strictly on a binary target output (0: Clean, 1: Debris Present) or micro-bbox anchor
#         self.fod_head = nn.Sequential(
#             nn.Conv2d(512, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(128, 1) # Binary classification logit for patch anomaly
#         )

#     def forward(self, x):
#         # FIX 2: Manually save intermediate layer tensors to handle YOLO skip connections
#         saved_features = []
        
#         # Route through the extracted backbone layers one by one safely
#         for i, layer in enumerate(self.shared_backbone):
#             # If your custom YOLO configuration needs specific layer lookbacks, 
#             # save them into the list here. For indices 0-9, they flow cleanly:
#             x = layer(x)
#             saved_features.append(x)
            
#         # At this point, index 9 output represents your shared backbone features
#         shared_features = x 

#         # Route identical feature maps to all 3 custom airport task heads simultaneously
#         out_turnaround = self.turnaround_head(shared_features)
#         out_ppe = self.ppe_head(shared_features)
#         out_fod = self.fod_head(shared_features)

#         return out_turnaround, out_ppe, out_fod

# class JointMultiTaskLoss(nn.Module):
#     def __init__(self, alpha=1.0, beta=1.5, gamma=2.0):
#         super(JointMultiTaskLoss, self).__init__()
#         self.alpha = alpha   # Turnaround weight
#         self.beta = beta     # PPE weight (boosted to penalize safety drops)
#         self.gamma = gamma   # FOD weight (highest weight because anomalies are small/sparse)
        
#         # Component loss functions
#         self.bbox_loss = nn.MSELoss()
#         self.cls_loss = nn.CrossEntropyLoss()
#         self.fod_binary_loss = nn.BCEWithLogitsLoss()

#     def forward(self, preds, targets):
#         pred_t, pred_p, pred_f = preds
#         tgt_t, tgt_p, tgt_f = targets

#         # Calculate isolated losses (simplified representation for demonstration)
#         loss_turnaround = self.bbox_loss(pred_t[:, :4], tgt_t[:, :4]) + self.cls_loss(pred_t[:, 4:], tgt_t[:, 4])
#         loss_ppe = self.bbox_loss(pred_p[:, :4], tgt_p[:, :4]) + self.cls_loss(pred_p[:, 4:], tgt_p[:, 4])
#         loss_fod = self.fod_binary_loss(pred_f, tgt_f)

#         # Unified backpropagation gradient vector
#         total_loss = (self.alpha * loss_turnaround) + (self.beta * loss_ppe) + (self.gamma * loss_fod)
#         return total_loss