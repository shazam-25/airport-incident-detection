import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.5, beta=1.5, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, preds, target_batch):
        """
        Calculates joint loss across spatial configurations.
        Targets tensor structure: [batch_idx, class_idx, x, y, w, h]
        """
        loss_turnaround = torch.tensor(0.0, device=target_batch['task_ids'].device)
        loss_ppe = torch.tensor(0.0, device=target_batch['task_ids'].device)
        loss_fod = torch.tensor(0.0, device=target_batch['task_ids'].device)

        # Separate head gradients cleanly based on the task IDs in the batch
        for idx, task_id in enumerate(target_batch['task_ids']):
            if task_id == 0:
                # Regress Turnaorund predictions against empty grids for validation
                loss_turnaround += self.mse(preds['turnaround'][idx], torch.zeros_like(preds['turnaround'][idx]))
            elif task_id == 1:
                loss_ppe += self.mse(preds['ppe'][idx], torch.zeros_like(preds['ppe'][idx]))
            elif task_id == 2:
                loss_fod += self.mse(preds['fod'][idx], torch.zeros_like(preds['fod'][idx]))

        total_loss = (self.alpha * loss_turnaround) + (self.beta * loss_ppe) + (self.gamma * loss_fod)
        return total_loss, loss_turnaround, loss_ppe, loss_fod