import sys
import os

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from common.data import load_data
    from common.preprocessing import get_preprocessor
    from common.models import get_model
    from common.cv import run_cv
    from common.submit import save_submission
    print("SUCCESS: usage: 'python scripts/verify_setup.py' - All common modules imported correctly.")
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
