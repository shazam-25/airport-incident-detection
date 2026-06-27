import torch
from models.model import MultiTaskAirportNet
from models.inference_sahi import sahi_fod_inference

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎬 Verifying Phase 2 SAHI Engine Pipeline on: {device}")

    # Initialize the model architecture
    model = MultiTaskAirportNet().to(device)

    # Simulate an incoming high-resolution runway stream frame (3 channels, 640x640)
    mock_runway_frame = torch.randn(3, 640, 640)
    
    try:
        boxes, scores, classes = sahi_fod_inference(model, mock_runway_frame, device)
        print("\n====== 🧪 PHASE 2 SAHI INFERENCE VERIFIED ======")
        print(f"  Processed high-detail patch transformations successfully.")
        print(f"  Detected Spatial Elements Tensor Shape: {boxes.shape}")
        print(f"  Confidence Scores Array Output Shape : {scores.shape}")
        print("=================================================")
        print("☑️ SAHI patch-processing logic successfully compiled!")
    except Exception as e:
        print(f"❌ Verification Failed: {e}")