import os
import argparse
from ultralytics import YOLO

def predict():
    parser = argparse.ArgumentParser(description='YOLO Inference Script')
    parser.add_argument('--source', type=str, required=True, help='Path to image or video')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (default: weights/yolov8n.pt)')
    parser.add_argument('--save', action='store_true', default=True, help='Save results to runs/detect')
    parser.add_argument('--show', action='store_true', help='Show results window')
    
    args = parser.parse_args()
    
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    # Determine weights path
    if args.weights:
        weights_path = args.weights
    else:
        # Default to yolov8n in weights dir
        weights_path = os.path.join(project_root, 'weights', 'yolov8n.pt')
        if not os.path.exists(weights_path):
             # Fallback to local file or download
             weights_path = 'yolov8n.pt'
    
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    
    # 2. Predict
    source_path = args.source
    
    # Check if source exists; if not, try to find it in the global data folder
    if not os.path.exists(source_path):
        # script_dir/../../../../data -> kaggle/data
        # script_dir is .../kaggle/competitions/yolo-playground/scripts
        # Global data is .../kaggle/data
        
        # Try finding it relative to project root's parent's parent (assuming standard structure)
        # project_root is .../yolo-playground
        # standard data path: .../kaggle/data
        
        # Determine global data path relative to script
        # script_dir: .../kaggle/competitions/yolo-playground/scripts
        # target:     .../kaggle/data
        # relative:   ../../../data
        
        candidate_path = os.path.normpath(os.path.join(script_dir, '../../../data', source_path))
        if os.path.exists(candidate_path):
            print(f"Source not found at '{source_path}', found at '{candidate_path}'")
            source_path = candidate_path
        else:
             # Also try just one level up if user moved things? No, standard is ../../../data
             pass

    print(f"Running inference on {source_path}...")
    results = model.predict(
        source=source_path,
        save=args.save,
        show=args.show,
        project=os.path.join(project_root, 'runs/detect'),
        name='predict'
    )
    
    print(f"Inference complete. Results saved to {os.path.join(project_root, 'runs/detect')}")

if __name__ == "__main__":
    predict()
