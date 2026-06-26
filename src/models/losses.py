import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.5, beta=1.5, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # SmmothL1 is standard for bounding box regression; CrossEntropy handles classes
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, preds, target_batch):
        """
        Calculates joint loss across spatial configurations.
        Targets tensor structure: [batch_idx, class_idx, x, y, w, h]
        """
        # Fallback trick: Find the first available GPU prediction tensor to read the device
        device = next(iter(preds.values())).device
        
        device = target_batch['task_ids'].device
        device = preds['turnaround'].device
        device = preds['ppe'].device
        device = preds['fod'].device

        # loss_turnaround = torch.tensor(0.0, device=target_batch['task_ids'].device)
        # loss_ppe = torch.tensor(0.0, device=target_batch['task_ids'].device)
        # loss_fod = torch.tensor(0.0, device=target_batch['task_ids'].device)
        loss_turnaround = torch.tensor(0.0, device=device)
        loss_ppe = torch.tensor(0.0, device=device)
        loss_fod = torch.tensor(0.0, device=device)

        packed_labels = target_batch['labels'].to(device)

        # Loop through each image in the batch to calculate true spatial differences
        for img_idx, task_id in enumerate(target_batch['task_ids']):
            # Filter out annotations belonging specifically to the single image
            img_mask = packed_labels[:, 0] == img_idx
            img_targets = packed_labels[img_mask]

            # If an image has no annotations, assign a small background target loss
            if img_targets.shape[0] == 0:
                continue

            # Extract true labels from data loader matrix
            true_classes = img_targets[:, 1].long()
            true_boxes = img_targets[:, 2:]

            if task_id == 0: # Turnaround Branch
                # 1. Grab the current GPU device dynamically
                # device = preds['turnaround'].device
                # Slice model output to extract bounding boxes vs class predictions
                pred_raw = preds['turnaround'][img_idx].mean(dim=[1, 2])    # Reduce spatial grid to match targets
                # 2. Push target tensors to the exact same GPU device
                box_target = true_boxes[0][:4].to(device)
                class_target = true_classes[0].unsqueeze(0).to(device)
                # 3. Calculate losses safely
                loss_turnaround += self.bbox_loss_fn(pred_raw[:4], box_target)
                loss_turnaround += self.cls_loss_fn(pred_raw[4:17].unsqueeze(0), class_target)

            elif task_id == 1: # PPE Branch
                # device = preds['ppe'].device
                pred_raw = preds['ppe'][img_idx].mean(dim=[1, 2])

                box_target = true_boxes[0][:4].to(device)
                class_target = true_classes[0].unsqueeze(0).to(device)

                loss_ppe += self.bbox_loss_fn(pred_raw[:4], box_target)
                loss_ppe += self.cls_loss_fn(pred_raw[4:8].unsqueeze(0), class_target)

            elif task_id == 2: # FOD Branch
                # device = preds['fod'].device
                pred_raw = preds['fod'][img_idx].mean(dim=[1, 2])

                box_target = true_boxes[0][:4].to(device)
                class_target = true_classes[0].unsqueeze(0).to(device)

                loss_fod += self.bbox_loss_fn(pred_raw[:4], box_target)
                loss_fod += self.cls_loss_fn(pred_raw[4:35].unsqueeze(0), class_target)
 
        # Separate head gradients cleanly based on the task IDs in the batch
        # for idx, task_id in enumerate(target_batch['task_ids']):
        #     if task_id == 0:
        #         # Regress Turnaorund predictions against empty grids for validation
        #         target_zeros = torch.zeros_like(preds['turnaround'][idx]).to(device)
        #         loss_turnaround += self.mse(preds['turnaround'][idx], target_zeros)
        #     elif task_id == 1:
        #         target_zeros = torch.zeros_like(preds['ppe'][idx]).to(device)
        #         loss_ppe += self.mse(preds['ppe'][idx], target_zeros)
        #         # loss_ppe += self.mse(preds['ppe'][idx], torch.zeros_like(preds['ppe'][idx]))
        #     elif task_id == 2:
        #         target_zeros = torch.zeros_like(preds['fod'][idx]).to(device)
        #         loss_fod += self.mse(preds['fod'][idx], target_zeros)
        #         # loss_fod += self.mse(preds['fod'][idx], torch.zeros_like(preds['fod'][idx]))

        total_loss = (self.alpha * loss_turnaround) + (self.beta * loss_ppe) + (self.gamma * loss_fod)
        return total_loss, loss_turnaround, loss_ppe, loss_fod