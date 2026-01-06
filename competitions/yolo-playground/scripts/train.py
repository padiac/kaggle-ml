import os
import yaml
import sys
from ultralytics import YOLO

def train():
    print("Starting YOLO Training...")
    
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    config_path = os.path.join(project_root, 'configs', 'yolo_baseline.yaml')
    data_yaml_path = os.path.join(project_root, 'configs', 'data_coco128.yaml')
    
    # 2. Load Config
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 3. Model
    # Try to load local weights first, else download
    model_type = config.get('model', {}).get('type', 'yolov8n')
    weights_path = os.path.join(project_root, 'weights', f"{model_type}.pt")
    
    if os.path.exists(weights_path):
        print(f"Loading local weights: {weights_path}")
        model = YOLO(weights_path)
    else:
        print(f"Local weights not found at {weights_path}. Downloading {model_type}.pt...")
        model = YOLO(f"{model_type}.pt")
        
    # 4. Train
    print("Starting training...")
    params = config.get('hyperparameters', {})
    
    # Ensure data points to our custom yaml
    # Note: Ultralytics requires absolute path for data yaml usually, or relative to cwd.
    # We pass absolute path to be safe.
    results = model.train(
        data=os.path.abspath(data_yaml_path),
        epochs=params.get('epochs', 10),
        batch=params.get('batch_size', 16),
        imgsz=params.get('img_size', 640),
        lr0=params.get('learning_rate', 0.01),
        project=os.path.join(project_root, 'runs'),
        name='exp'
    )
    
    print("Training complete!")

if __name__ == "__main__":
    train()
