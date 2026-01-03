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
    train_df, test_df = load_data(config['competition_name'])
    
    # 2. Prepare Data
    target_col = config['target_column']
    id_col = config['id_column']
    
    X = train_df.drop(columns=[target_col, id_col])
    y = train_df[target_col]
    
    # Test data has ID_code and features
    test_ids = test_df[id_col]
    X_test = test_df.drop(columns=[id_col])
    
    # --- Auto-detect Features (var_0 to var_199) ---
    numeric_cols = [c for c in X.columns if c.startswith('var_')]
    categorical_cols = []
    
    print(f"Detected {len(numeric_cols)} numeric features (var_0..var_199).")
    
    # Update config for transparency/logging
    config['features']['numeric'] = numeric_cols
    
    # Select only relevant columns
    X = X[numeric_cols]
    X_test = X_test[numeric_cols]
    
    # 3. Preprocessing
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
    # For ROC-AUC, we often want probabilities, but save_submission usually takes classes or regression values.
    # If the competition requires probability (which Santander does), we should use predict_proba.
    # However, common/submit.py is generic. Let's check common/models.py or assume predict for now,
    # but for AUC we definitely want probabilities.
    
    if hasattr(pipeline, "predict_proba"):
        # predict_proba returns [n_samples, n_classes], we want probability of class 1
        preds = pipeline.predict_proba(X_test)[:, 1]
    else:
        preds = pipeline.predict(X_test)
    
    # 8. Submit
    save_submission(
        ids=test_ids, 
        preds=preds, 
        id_col=id_col, 
        target_col=target_col,
        filename="submission.csv"
    )

if __name__ == "__main__":
    main()
