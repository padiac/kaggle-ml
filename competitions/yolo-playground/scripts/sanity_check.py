import torch
import sys
import os

def check_environment():
    print("--- YOLO Playground Sanity Check ---")
    
    # Check Python
    print(f"Python: {sys.version.split()[0]}")
    
    # Check PyTorch
    try:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        
    # Check YOLO (ultralytics)
    try:
        import ultralytics
        print(f"Ultralytics (YOLO): {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics: NOT INSTALLED (Run 'pip install ultralytics')")

    print("--- Check Complete ---")

if __name__ == "__main__":
    check_environment()
