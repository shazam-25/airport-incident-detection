import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import AirportMultiTaskDataset, multi_task_collate_fn
from models.model import MultiTaskAirportNet
from models.losses import MultiTaskLoss

# GradScaler handles lower precision gradients smoothly
scaler = torch.amp.GradScaler('cuda')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_l_t = 0.0
    running_l_p = 0.0
    running_l_f = 0.0

    pbar = tqdm(dataloader, desc="  Training Progress")
    for batch in pbar:
        images = batch['images'].to(device)
        optimizer.zero_grad()

        # 🟢 FIX 1: Wrap forward pass in mixed precision context without duplicate calls
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred = model(images)
            total_loss, l_t, l_p, l_f = criterion(pred, batch)

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
            # 🟢 FIX 2: Wrapped validation loop in autocast and removed duplicate calls
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(images)
                total_loss, _, _, _ = criterion(preds, batch)
            running_loss += total_loss.item()
    
    return running_loss / num_batches


def model_train_phase1(proc_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Full-Scale Training Pipeline on Target: {device}")

    current_script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.abspath(os.path.join(current_script_dir, "../../")) 
    
    target_log_dir = os.path.abspath(os.path.join(project_root, "models", "logs", "phase1")) 
    os.makedirs(target_log_dir, exist_ok=True) 

    writer = SummaryWriter(log_dir=target_log_dir)

    # 1. Ingest Full Datasets
    train_dataset = AirportMultiTaskDataset(root_dir=proc_dir, split="train", img_size=640)
    val_dataset = AirportMultiTaskDataset(root_dir=proc_dir, split="val", img_size=640)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)

    # 2. Setup Network and Optimization Metrics
    model = MultiTaskAirportNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    criterion = MultiTaskLoss(alpha=1.5, beta=1.5, gamma=0.5)

    best_val_loss = float('inf')
    epochs = 20 

    for epoch in range(1, epochs + 1):
        print(f"\n📅 Epoch [{epoch}/{epochs}]")

        train_loss, l_t, l_p, l_f = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"    📊 Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    └─ Heads Breakdown -> Turnaround: {l_t:.4f} | PPE: {l_p:.4f} | FOD: {l_f:.4f}")

        writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Head_Turnaround", l_t, epoch)
        writer.add_scalar("Loss/Head_PPE", l_p, epoch)
        writer.add_scalar("Loss/Head_FOD", l_f, epoch)

        # Save weights file checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_weights_repo = os.path.abspath(os.path.join(project_root, "models", "weights"))
            os.makedirs(model_weights_repo, exist_ok=True)
            checkpoint_path = os.path.join(model_weights_repo, "checkpoint_v1.pt") 
            torch.save(model.state_dict(), checkpoint_path)
            print(" 💾 New optimal validation milestone secured. Checkpoint saved.")
    
    writer.close()
    print("\n☑️ Training Sequence Concluded Successfully.")


def model_train_phase2(proc_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Phase 2: Targeted FOD Fine-Tuning on Target: {device}")

    current_script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.abspath(os.path.join(current_script_dir, "../../")) 
    
    target_log_dir = os.path.abspath(os.path.join(project_root, "models", "logs", "phase2"))
    os.makedirs(target_log_dir, exist_ok=True) 

    writer = SummaryWriter(log_dir=target_log_dir)

    # 1. Ingest Full Datasets
    train_dataset = AirportMultiTaskDataset(root_dir=proc_dir, split="train", img_size=640)
    val_dataset = AirportMultiTaskDataset(root_dir=proc_dir, split="val", img_size=640)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)

    # 2. Setup Network and Restore Phase 1 Pre-trained Weights
    model = MultiTaskAirportNet().to(device)
    # 🟢 FIX 3: Fixed the folder restoration path to look inside models/weights/
    phase1_weights_path = os.path.abspath(os.path.join(project_root, "models", "weights", "checkpoint_v1.pt"))

    if os.path.exists(phase1_weights_path):
        model.load_state_dict(torch.load(phase1_weights_path, map_location=device))
        print("💾 Phase 1 Base Weights Successfully Restored.")
    else:
        print("⚠️ Warning: Phase 1 weights checkpoint file not found. Running baseline setup instead.")

    # 3. Structural Parameter Freezing
    print("🔒 Locking Down Turnaround and PPE Branch Weights...")
    for param in model.turnaround_head.parameters():
        param.requires_grad = False
    for param in model.ppe_head.parameters():
        param.requires_grad = False
    
    for param in model.fod_head.parameters():
        param.requires_grad = True
    
    # 4. Hyperparameter Adaptation for Micro-Object Optimization
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 🟢 COMPLETE: Configured Inverted Coefficients Matrix (FOD focus)
    criterion = MultiTaskLoss(alpha=0.0, beta=0.0, gamma=2.0)

    best_val_loss = float('inf')
    epochs = 10  

    # 🟢 COMPLETE: Phase 2 Execution Loop and Metric Logging Block
    for epoch in range(1, epochs + 1):
        print(f"\n📅 Phase 2 - Epoch [{epoch}/{epochs}]")

        train_loss, l_t, l_p, l_f = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"    📊 Phase 2 Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    └─ Isolated Head Focus -> FOD Loss: {l_f:.4f}")

        writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Head_FOD", l_f, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_weights_repo = os.path.abspath(os.path.join(project_root, "models", "weights"))
            checkpoint_path = os.path.join(model_weights_repo, "checkpoint_v2.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(" 💾 Phase 2 optimal milestone secured. Checkpoint saved.")

    writer.close()
    print("\n☑️ Phase 2 Fine-Tuning Sequence Concluded Successfully.")
