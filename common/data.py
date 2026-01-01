import os
import glob
import pandas as pd

def load_data(competition_name, keyword=None):
    """
    Load train and test data for a given competition.
    Auto-detects environment (Kaggle vs Local).
    
    Args:
        competition_name (str): Name of the competition (folder name).
        keyword (str, optional): Keyword to identify dataset folder in Kaggle input.
        
    Returns:
        tuple: (train_df, test_df)
    """
    
    # Check for Kaggle environment
    kaggle_input_path = "/kaggle/input"
    is_kaggle = os.path.exists(kaggle_input_path)
    
    data_dir = None
    
    if is_kaggle:
        print(f"Detected Kaggle environment per {kaggle_input_path}")
        # Search for the dataset folder
        # Usually /kaggle/input/{competition_name} or /kaggle/input/{keyword}
        
        candidates = glob.glob(os.path.join(kaggle_input_path, "*"))
        
        # Filter candidates if keyword provided
        if keyword:
            candidates = [c for c in candidates if keyword in c]
            
        if not candidates:
            # Fallback: maybe the folder is exactly the competition name/slug
            direct_path = os.path.join(kaggle_input_path, competition_name)
            if os.path.exists(direct_path):
                data_dir = direct_path
            else:
                raise FileNotFoundError(f"Could not locate data directory in {kaggle_input_path} with keyword '{keyword}'")
        elif len(candidates) == 1:
            data_dir = candidates[0]
        else:
            # Multiple candidates found, try to find one that contains 'train.csv'
            for c in candidates:
                if os.path.exists(os.path.join(c, "train.csv")):
                    data_dir = c
                    break
            
            if not data_dir:
                 # If still ambiguous, warn and pick the first one or error
                print(f"Warning: Multiple dataset candidates found: {candidates}. Using {candidates[0]}")
                data_dir = candidates[0]
                
    else:
        # Local environment
        # Assumption: run from the competition folder or root
        # We expect data to be in ../../data/{competition_name} relative to competition folder
        # OR ./data/{competition_name} relative to root
        
        possible_roots = [
            os.path.join(os.getcwd(), "data", competition_name), # Run from root
            os.path.abspath(os.path.join(os.getcwd(), "../../data", competition_name)), # Run from comp folder
            os.path.join(os.path.dirname(__file__), "../data", competition_name) # Run from package
        ]
        
        for root in possible_roots:
            if os.path.exists(root) and os.path.exists(os.path.join(root, "train.csv")):
                data_dir = root
                break
                
        if not data_dir:
            raise FileNotFoundError(f"Could not locate data for '{competition_name}'. Checked: {possible_roots}")

    print(f"Loading data from: {data_dir}")
    
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {data_dir}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found in {data_dir}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    return train_df, test_df
