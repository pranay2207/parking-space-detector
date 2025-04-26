#!/usr/bin/env python3
import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    # Get current working directory
    cwd = Path.cwd()
    
    # Path to data.yaml
    data_path = cwd / 'data.yaml'
    
    # Path to save models
    models_dir = cwd / 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data.yaml to get dataset info
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Training on dataset with {data_config['nc']} classes: {data_config['names']}")
    print(f"Training data path: {data_config['train']}")
    print(f"Validation data path: {data_config['val']}")
    
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model - use shorter training for quick results
    results = model.train(
        data=str(data_path),
        epochs=50,  # Reduced for faster training
        imgsz=640,
        project=str(models_dir),
        name='parking_detector',
        batch=16,
        patience=20,
        save=True,
        verbose=True
    )
    
    # Validate the model
    val_results = model.val()
    print(f"Validation results: mAP50 = {val_results.box.map50:.4f}, "
          f"Precision = {val_results.box.precision:.4f}, "
          f"Recall = {val_results.box.recall:.4f}")
    
    # Export the model for inference
    model.export(format="onnx")
    
    print(f"Model training complete. Model saved to {models_dir / 'parking_detector'}")
    print(f"ONNX model saved to {models_dir / 'parking_detector' / 'weights' / 'best.onnx'}")

if __name__ == "__main__":
    main() 