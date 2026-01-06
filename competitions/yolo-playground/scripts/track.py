import os
import argparse
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def track():
    parser = argparse.ArgumentParser(description='YOLO Object Tracking Script')
    parser.add_argument('--source', type=str, required=True, help='Path to video file or image folder')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (default: weights/yolov8n.pt)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker config (bytetrack.yaml or botsort.yaml)')
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
             weights_path = 'yolov8n.pt'
    
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    # 2. Source Resolution
    source_path = args.source
    if not os.path.exists(source_path):
        # script_dir: .../kaggle/competitions/yolo-playground/scripts
        # target:     .../kaggle/data
        # relative:   ../../../data
        data_root = os.path.normpath(os.path.join(script_dir, '../../../data'))
        candidate_path = os.path.join(data_root, source_path)
        
        if os.path.exists(candidate_path):
            print(f"Source found at '{candidate_path}'")
            source_path = candidate_path
        else:
            # Check for double-nested directory (common in Kaggle datasets, e.g. MOT17/MOT17)
            # If source starts with "MOT17", check "data/MOT17/MOT17/..."
            parts = source_path.replace('\\', '/').split('/')
            if len(parts) > 0:
                nested_candidate = os.path.join(data_root, parts[0], source_path)
                if os.path.exists(nested_candidate):
                    print(f"Source found at nested path '{nested_candidate}'")
                    source_path = nested_candidate

    # 3. Handle Directory (Image Sequence)
    if os.path.isdir(source_path):
        print(f"Detected image folder: {source_path}")
        print("Generating temporary video file...")
        
        # Output directory for temp video
        # We'll put it in runs/detect alongside the tracking results, or just runs/detect
        # Let's put it in runs/detect/temp to be safe/clean
        temp_dir = os.path.join(project_root, 'runs', 'detect', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Video name based on folder name
        folder_name = os.path.basename(os.path.normpath(source_path))
        video_path = os.path.join(temp_dir, f"{folder_name}.mp4")
        
        # Get images
        # Try common extensions
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for f in os.listdir(source_path):
            if os.path.splitext(f)[1].lower() in exts:
                images.append(f)
        
        images.sort()
        
        if not images:
            print(f"Error: No images found in {source_path}")
            return

        # Read first image
        first_img = cv2.imread(os.path.join(source_path, images[0]))
        height, width, _ = first_img.shape
        
        # Writer
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for img_name in tqdm(images, desc="Stitching video"):
            img_path = os.path.join(source_path, img_name)
            frame = cv2.imread(img_path)
            out.write(frame)
        
        out.release()
        print(f"Video generated at: {video_path}")
        
        # Update source_path to the new video
        source_path = video_path

    # 4. Track
    print(f"Running tracking on {source_path} using {args.tracker}...")
    results = model.track(
        source=source_path,
        tracker=args.tracker,
        save=args.save,
        show=args.show,
        project=os.path.join(project_root, 'runs/detect'),
        name='track'
    )
    
    print(f"Tracking complete. Results saved to {os.path.join(project_root, 'runs/detect')}")

if __name__ == "__main__":
    track()
