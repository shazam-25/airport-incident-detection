import os
from ultralytics import YOLO

def train_incident_model(data_yaml_path, model_variant='yolov8n.pt', epochs=100, imgsz=640, device=0):
    """
    Trains the YOLO model on pre-split train/val sets.
    Saves outputs to the models directory.
    """
    print(f"\n--- Initializing Training with {model_variant} ---")
    model = YOLO(model_variant)

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project='models',       # Saves results under models/ directory
        name='incident_train',   # Creates models/incident_train/ folder
        save=True
    )

    best_weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"Training Complete. Model weights saved to: {best_weights_path}")

    return best_weights_path
