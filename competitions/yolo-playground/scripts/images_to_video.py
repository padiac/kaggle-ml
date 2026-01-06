import cv2
import os
import argparse
from tqdm import tqdm

def images_to_video():
    parser = argparse.ArgumentParser(description='Convert image sequence to video')
    parser.add_argument('--source', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Output video path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--ext', type=str, default='jpg', help='Image extension (jpg, png, etc.)')
    
    args = parser.parse_args()
    
    # Resolve source path (using same logic as predict/track)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = args.source
    
    if not os.path.exists(source_path):
        data_root = os.path.normpath(os.path.join(script_dir, '../../../data'))
        candidate_path = os.path.join(data_root, source_path)
        if os.path.exists(candidate_path):
            source_path = candidate_path
        else:
             parts = source_path.replace('\\', '/').split('/')
             if len(parts) > 0:
                nested_candidate = os.path.join(data_root, parts[0], source_path)
                if os.path.exists(nested_candidate):
                    source_path = nested_candidate
    
    if not os.path.exists(source_path):
        print(f"Error: Source directory '{source_path}' not found.")
        return

    print(f"Reading images from {source_path}...")
    
    images = [img for img in os.listdir(source_path) if img.endswith(args.ext)]
    images.sort() # Ensure correct order
    
    if not images:
        print(f"No images found with extension .{args.ext} in {source_path}")
        return

    # Read first image to get dimensions
    first_image_path = os.path.join(source_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure mp4v works, or try 'XVID' for .avi
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    print(f"Creating video '{args.output}' ({width}x{height} @ {args.fps}fps)...")
    
    for image_name in tqdm(images):
        image_path = os.path.join(source_path, image_name)
        frame = cv2.imread(image_path)
        out.write(frame)

    out.release()
    print("Video generation complete!")

if __name__ == "__main__":
    images_to_video()
