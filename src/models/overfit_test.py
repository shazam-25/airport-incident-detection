import torch
import torch.optim as optim
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.dataset import AirportMultiTaskDataset, multi_task_collate_fn
from models.model import MultiTaskAirportNet
from models.losses import MultiTaskLoss

def run_overfitting_check(proc_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔋 Execution Target Device: {device}")

    # 1. Ingest Data Loader
    dataset = AirportMultiTaskDataset(root_dir=proc_dir, split="train", img_size=640)
    # loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=multi_task_collate_fn)

    # Balanced Sampler Logic
    # Force find exactly one sample from each task track
    samples_by_task = {0: None, 1: None, 2: None}
    for idx in range(len(dataset)):
        # sample = dataset[idx]
        t_id = int(dataset.samples[idx]['task_id'])
        if samples_by_task[t_id] is None:
            samples_by_task[t_id] = dataset[idx]
        if all(v is not None for v in samples_by_task.values()):
            break
    
    # Build a controlled batch of exactly 3 items (one per task) using collate routine
    balanced_list = [samples_by_task[0], samples_by_task[1], samples_by_task[2]]
    fixed_batch = multi_task_collate_fn(balanced_list)

    # Ship to execution hardware
    fixed_batch['images'] = fixed_batch['images'].to(device)
    fixed_batch['task_ids'] = fixed_batch['task_ids'].to(device)

    # 2. Initialize Network & Loss
    model = MultiTaskAirportNet().to(device)
    criterion = MultiTaskLoss(alpha=1.5, beta=1.5, gamma=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 3. History Trackers for Data Presentation PLots
    history_total = []
    history_turnaround = []
    history_ppe = []
    history_fod = []

    print("\n🚀 Starting Overfitting Test Run (50 Epochs)...")
    model.train()
    for epoch in range(1, 51):
        optimizer.zero_grad()
        preds = model(fixed_batch['images'])

        total_loss, l_t, l_p, l_f = criterion(preds, fixed_batch)
        total_loss.backward()
        optimizer.step()

        # Append telemetry records safely to host memory
        history_total.append(total_loss.item())
        history_turnaround.append(l_t.item())
        history_ppe.append(l_p.item())
        history_fod.append(l_f.item())

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch:02d}/50] -> Total Loss: {total_loss.item():.6f} | "
                  f"Turnaround Loss: {l_t.item():.4f} | PPE Loss: {l_p.item():.4f} | FOD Loss: {l_f.item():.4f}")

    # print("\n=================================================================================")
    if total_loss.item() < 0.1:
        print("☑️ OVERFITTING TEST PASSED: Model safely optimized small sample batch vector!")
    else:
        print("⚠️ Warning: Optimization error. Check learning rates or parameters.")
    print("==================================================================================\n")

    # ========================================================
    # 4. PLOTTING TELEMETRY PROFILE FOR LESSON REPORT
    # ========================================================
    epochs_range = range(1, 51)
    plt.figure(figsize=(10, 5))
    
    # Plot Total Loss line profile
    plt.plot(epochs_range, history_total, label='Total Joint Loss', color='black', linewidth=2.5, linestyle='--')
    
    # Plot Individual Head Convergence Lines
    plt.plot(epochs_range, history_turnaround, label='Head 1: Turnaround Loss', color='royalblue', linewidth=1.8)
    plt.plot(epochs_range, history_ppe, label='Head 2: PPE Compliance Loss', color='darkorange', linewidth=1.8)
    plt.plot(epochs_range, history_fod, label='Head 3: FOD Anomaly Loss', color='crimson', linewidth=1.8)
    
    # Figure Formatting Details
    plt.title("Multi-Task Network Model Convergence Profile (Overfitting Sanity Check)", fontsize=12, fontweight='bold')
    plt.xlabel("Optimization Training Epoch Matrices", fontsize=10)
    plt.ylabel("Computed Objective Loss Metrics", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    # Save chart to weights/ directory layout before popping window display
    # plt.savefig("./reports/overfit_convergence_profile.png", dpi=300)
    # print("💾 Plot metrics chart safely exported to: ./reports/overfit_convergence_profile.png")
    plt.show()


# def run_overfitting_test():
#     print("=== 🧪 Starting Multi-Task Overfitting Sanity Check ===")
    
#     # Force execution device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using execution device: {device}")

#     # 1. Initialize Pipeline components
#     dataset = AirportMultiTaskDataset(root_dir="/teamspace/studios/this_studio/airport-incident-detection/data/processed", split="train")
#     loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=multi_task_collate_fn)
    
#     # Pull exactly ONE batch and lock it in memory
#     images, labels, task_ids = next(iter(loader))
#     images = images.to(device)
#     task_ids = task_ids.to(device)
    
#     # 2. Instantiate Model and Optimizer
#     model = MultiHeadAirportYOLO().to(device)
#     model.train()  # Explicitly set network layer to training mode
    
#     # Using AdamW with a slightly higher learning rate for fast memorization
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
#     # Simplified Dummy Joint Loss for Sanity Check validation
#     # (Matches Week 2 custom loss behavior)
#     criterion_mse = torch.nn.MSELoss()
#     criterion_bce = torch.nn.BCEWithLogitsLoss()

#     print("\n🚀 Training model on a single batch for 50 epochs...")
#     print("-" * 50)
    
#     for epoch in range(1, 51):
#         optimizer.zero_grad()
        
#         # Forward Pass through all 3 heads
#         out_turnaround, out_ppe, out_fod = model(images)
        
#         # Calculate isolated mock losses against target shapes
#         # (Using mean activations to track convergence for this structural check)
#         loss_t = criterion_mse(out_turnaround.mean(dim=[2, 3]), torch.zeros((4, 17), device=device))
#         loss_p = criterion_mse(out_ppe.mean(dim=[2, 3]), torch.zeros((4, 8), device=device))
#         loss_f = criterion_bce(out_fod.squeeze(), task_ids.float())
        
#         # Joint Multi-Task Loss execution
#         total_loss = loss_t + loss_p + loss_f
        
#         # Backpropagation step
#         total_loss.backward()
#         optimizer.step()
        
#         # Print logs every 10 epochs to monitor drop progress
#         if epoch == 1 or epoch % 10 == 0:
#             print(f"Epoch [{epoch:02d}/50] -> Total Joint Loss: {total_loss.item():.6f} | "
#                   f"Turnaround: {loss_t.item():.4f} | PPE: {loss_p.item():.4f} | FOD: {loss_f.item():.4f}")

#     print("-" * 50)
#     if total_loss.item() < 0.1:
#         print("✅ SUCCESS: The joint loss converged close to 0! Your network is architecturally sound.")
#     else:
#         print("❌ FAILED: Loss stalled. Double-check your head channels or loss calculations.")

# if __name__ == "__main__":
#     run_overfitting_test()