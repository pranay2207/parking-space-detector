#!/usr/bin/env python3
import os
import re
import sys

def update_app_with_new_model_path():
    # New model path
    new_model_path = "models/parking_detector/weights/best.pt"
    
    # Check if source file exists
    app_file_path = "src/app.py"
    if not os.path.exists(app_file_path):
        print(f"Error: Application file not found at {os.path.abspath(app_file_path)}")
        return
    
    # Check if model directory exists, create if not
    model_dir = os.path.dirname(new_model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created directory: {os.path.abspath(model_dir)}")
    
    # Print warning if model file doesn't exist yet
    if not os.path.exists(new_model_path):
        print(f"Warning: Trained model not found at {os.path.abspath(new_model_path)}")
        print("The app will be updated to use this path, but you'll need to train the model first")
    
    try:
        # Read the current content of app.py
        with open(app_file_path, "r") as f:
            content = f.read()
        
        # Define patterns to match model loading lines
        patterns = [
            # Match yolov8n.pt
            (r'model\s*=\s*YOLO\([\'"]yolov8n\.pt[\'"]\)', f'model = YOLO("{new_model_path}")'),
            # Match models/latest_model.pt
            (r'model\s*=\s*YOLO\([\'"]models/latest_model\.pt[\'"]\)', f'model = YOLO("{new_model_path}")'),
            # Match any YOLO model loading
            (r'model\s*=\s*YOLO\([\'"][^\'"]+[\'"]\)', f'model = YOLO("{new_model_path}")'),
        ]
        
        # Apply the replacements
        updated = False
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                updated = True
                break
        
        if not updated:
            print("Warning: Could not find a model loading line to update.")
            print("The app.py file may not be using the expected format.")
            return
        
        # Write the updated content back to app.py
        with open(app_file_path, "w") as f:
            f.write(content)
        
        print(f"Successfully updated {app_file_path} to use the new model at {new_model_path}")
        
    except Exception as e:
        print(f"Error updating app.py: {str(e)}")

if __name__ == "__main__":
    update_app_with_new_model_path() 