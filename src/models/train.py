import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import AirportMultiTaskDataset, multi_task_collate_fn
from models.model import MultiTaskAirportNet
from models.losses import MultiTaskLoss

# 🟢 Add GradScaler to handle lower precision gradients smoothly
scaler = torch.amp.GradScaler('cuda')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_l_t = 0.0
    running_l_p = 0.0
    running_l_f = 0.0

    pbar = tqdm(dataloader, desc="  Training Progress")
    for batch in pbar:
        # if short_sanity_check and batch_idx >= 5:
        #     print(f"\n🎯 Sanity Check: Truncating epoch early at {batch_idx} batches.")
        #     break

        images = batch['images'].to(device)

        optimizer.zero_grad()
        pred = model(images)

        total_loss, l_t, l_p, l_f = criterion(pred, batch)

        # Guard against zero-loss backward passes if a task is missing from the batch
        # total_loss.backward()
        # optimizer.step()
        # 🟢 Use scaler to handle backward pass and step optimization
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        running_l_t += l_t.item()
        running_l_p += l_p.item()
        running_l_f += l_f.item()

        pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})

    num_batches = len(dataloader)
    return (running_loss / num_batches,
            running_l_t / num_batches,
            running_l_p / num_batches,
            running_l_f / num_batches)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            preds = model(images)
            total_loss, _, _, _ = criterion(preds, batch)
            running_loss += total_loss.item()
    
    return running_loss / num_batches


def model_train_phase1(proc_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Full-Scale Training Pipeline on Target: {device}")

    # Define path dynamically relative to this script to guarantee it hits root/models/logs
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # points to src/models/
    project_root = os.path.abspath(os.path.join(current_script_dir, "../../")) # moves up to project root
    target_log_dir = os.path.join(project_root, "models", "logs")
    os.makedirs(target_log_dir, exist_ok=True) # Ensure folder exists safely

    # Initialize TensorBoard Logger
    writer = SummaryWriter(log_dir=target_log_dir)

    # 1. Ingest Full Datasets
    train_dataset = AirportMultiTaskDataset(root_dir=proc_dir, split="train", img_size=640)
    val_dataset = AirportMultiTaskDataset(root_dir=proc_dir, split="val", img_size=640)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)

    # 2. Setup Network and Optimization Metrics (Phase 1 Settings)
    model = MultiTaskAirportNet().to(device)
    # Compile model graph to unlock faster kernel execution paths
    # model = torch.compile(model) 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Loss coefficients configured to counter class balancing discrepencies
    criterion = MultiTaskLoss(alpha=1.5, beta=1.5, gamma=0.5) # change beta = 3.5 & gamma = 0.1

    best_val_loss = float('inf')
    epochs = 5 # Original - 15

    for epoch in range(1, epochs + 1):
        print(f"\n📅 Epoch [{epoch}/{epochs}]")

        # Execute training and validation loops
        train_loss, l_t, l_p, l_f = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"    📊 Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    └─ Heads Breakdown -> Turnaround: {l_t:.4f} | PPE: {l_p:.4f} | FOD: {l_f:.4f}")

        # Log values to TensorBoard for evaluation profiles
        writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Head_Turnaround", l_t, epoch)
        writer.add_scalar("Loss/Head_PPE", l_p, epoch)
        writer.add_scalar("Loss/Head_FOD", l_f, epoch)

        # Save weights file checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(project_root, "models", "checkpoint_v1.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(" 💾 New optimal validation milestone secured. Checkpoint saved.")
    
    writer.close()
    print("\n☑️ Training Sequence Concluded Successfully.")


# def model_train_sanity_fn(proc_dir):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🚀 Initializing Full-Scale Training Pipeline on Target: {device}")

#     RUN_SANITY_CHECK = True

