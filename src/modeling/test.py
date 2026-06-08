import os
import cv2
import time
import queue
import threading
import yt_dlp
from ultralytics import YOLO

def evaluate_test_performance(model_path, data_yaml_path):
    """
    Explicitly evaluate the trained best.pt model on the 'test' split.
    Prints model metric logging.
    """
    print(f"\n--- Evaluating Model Performance on Test Split ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    
    model = YOLO(model_path)

    # Run validation forced on the standalone test split
    metrics = model.val(
        data=data_yaml_path,
        split='test',
        projects='models',
        name='incident_test'    # Saves plots and metrics to model to models/incident_test/
    )

    # Extract keys safely from results dictionary
    precision = metrics.results_dict.get('metrics/precision(B)', 0)
    recall = metrics.results_dict.get('metrics/recall(B)', 0)
    map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)

    print("\n" + "="*45)
    print(f"{'TEST METRIC':<25} | {'VALUSE':<10}")
    print("="*45)
    print(f"{'Precision':<25} | {precision:.4f}")
    print(f"{'Recall':<25} | {recall:.4f}")
    print(f"{'mAP @ 0.5':<25} | {map50:.4f}")
    print(f"{'mAP @ 0.5:0.95':<25} | {map50_95:.4f}")
    print("="*45 + "\n")

    print(f"Confusion Matrix & Evaluation plots generated in: models/incident_test/")
    return metrics.save_dir