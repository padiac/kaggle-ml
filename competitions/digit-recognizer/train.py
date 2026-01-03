import sys
import os
import yaml
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from common.data import load_data
from common.preprocessing import get_preprocessor
from common.models import get_model
from common.cv import run_cv
from common.submit import save_submission

def main():
    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Starting {config['competition_name']} - {config['task']}")
    
    # 1. Load Data
    # keyword argument helps if the folder name is different or ambiguous
    train_df, test_df = load_data(config['competition_name'])
    
    # 2. Prepare Data
    target_col = config['target_column']
    
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    X_test = test_df.copy() # Test data usually just has pixels
    
    # --- Auto-detect Features (Pixels) ---
    # Digit Recognizer has pixel0 to pixel783
    numeric_cols = [c for c in X.columns if c.startswith('pixel')]
    categorical_cols = []
    
    # Update config for logging/record keeping if needed
    config['features']['numeric'] = numeric_cols
    
    print(f"Detected {len(numeric_cols)} numeric features (pixels).")
    
    # Select only relevant columns
    X = X[numeric_cols]
    X_test = X_test[numeric_cols]
    
    # 3. Preprocessing
    # We use the standard get_preprocessor. 
    # For Images, scaling is important. StandardScaler or MinMax (0-1) is good.
    # The current pipelines uses StandardScaler which is fine for RF/SVM.
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    
    # 4. Model
    model = get_model(config['task'], config['model']['name'], config['model'].get('params', {}))
    
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # 5. Cross Validation
    run_cv(
        pipeline, 
        X, 
        y, 
        task=config['task'], 
        n_splits=config['cv']['n_splits'], 
        scoring=config['cv']['scoring']
    )
    
    # 6. Train on Full Data
    print("Training on full dataset...")
    pipeline.fit(X, y)
    
    # 7. Predict
    print("Predicting on test set...")
    preds = pipeline.predict(X_test)
    
    # 8. Submit
    # Digit Recognizer usually requires ImageId (1-based index)
    if config['id_column'] in test_df.columns:
        test_ids = test_df[config['id_column']]
    else:
        print("Generating IDs for submission (1-based index)...")
        test_ids = pd.Series(range(1, len(X_test) + 1), name=config['id_column'])

    save_submission(
        ids=test_ids, 
        preds=preds, 
        id_col=config['id_column'], 
        target_col=target_col,
        filename="submission.csv"
    )

if __name__ == "__main__":
    main()
