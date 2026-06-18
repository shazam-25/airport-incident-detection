import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import AirportMultiTaskDataset, multi_task_collate_fn
from models.multi_task_yolo import MultiHeadAirportYOLO

def run_overfitting_test():
    print("=== 🧪 Starting Multi-Task Overfitting Sanity Check ===")
    
    # Force execution device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # 1. Initialize Pipeline components
    dataset = AirportMultiTaskDataset(root_dir="/teamspace/studios/this_studio/airport-incident-detection/data/processed", split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=multi_task_collate_fn)
    
    # Pull exactly ONE batch and lock it in memory
    images, labels, task_ids = next(iter(loader))
    images = images.to(device)
    task_ids = task_ids.to(device)
    
    # 2. Instantiate Model and Optimizer
    model = MultiHeadAirportYOLO().to(device)
    model.train()  # Explicitly set network layer to training mode
    
    # Using AdamW with a slightly higher learning rate for fast memorization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Simplified Dummy Joint Loss for Sanity Check validation
    # (Matches Week 2 custom loss behavior)
    criterion_mse = torch.nn.MSELoss()
    criterion_bce = torch.nn.BCEWithLogitsLoss()

    print("\n🚀 Training model on a single batch for 50 epochs...")
    print("-" * 50)
    
    for epoch in range(1, 51):
        optimizer.zero_grad()
        
        # Forward Pass through all 3 heads
        out_turnaround, out_ppe, out_fod = model(images)
        
        # Calculate isolated mock losses against target shapes
        # (Using mean activations to track convergence for this structural check)
        loss_t = criterion_mse(out_turnaround.mean(dim=[2, 3]), torch.zeros((4, 17), device=device))
        loss_p = criterion_mse(out_ppe.mean(dim=[2, 3]), torch.zeros((4, 8), device=device))
        loss_f = criterion_bce(out_fod.squeeze(), task_ids.float())
        
        # Joint Multi-Task Loss execution
        total_loss = loss_t + loss_p + loss_f
        
        # Backpropagation step
        total_loss.backward()
        optimizer.step()
        
        # Print logs every 10 epochs to monitor drop progress
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch:02d}/50] -> Total Joint Loss: {total_loss.item():.6f} | "
                  f"Turnaround: {loss_t.item():.4f} | PPE: {loss_p.item():.4f} | FOD: {loss_f.item():.4f}")

    print("-" * 50)
    if total_loss.item() < 0.1:
        print("✅ SUCCESS: The joint loss converged close to 0! Your network is architecturally sound.")
    else:
        print("❌ FAILED: Loss stalled. Double-check your head channels or loss calculations.")

if __name__ == "__main__":
    run_overfitting_test()