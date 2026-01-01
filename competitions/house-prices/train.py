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
    # For house prices, the slug might be 'house-prices-advanced-regression-techniques'
    # We pass the full name from config
    train_df, test_df = load_data(config['competition_name'])
    
    X = train_df.drop(columns=[config['target_column']])
    y = train_df[config['target_column']]
    X_test = test_df.copy()
    
    # --- OPTIONAL: Log Transform Target ---
    use_log_transform = config.get('log_transform_target', False)
    if use_log_transform:
        print("Applying np.log1p to target variable...")
        y = np.log1p(y)
    # --------------------------------------
    
    numeric_cols = config['features']['numeric']
    categorical_cols = config['features']['categorical']
    
    test_ids = test_df[config['id_column']]
    
    X = X[numeric_cols + categorical_cols]
    X_test = X_test[numeric_cols + categorical_cols]
    
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
    preds = pipeline.predict(X_test)
    
    # --- INVERSE TRANSFORM ---
    if use_log_transform:
        print("Applying np.expm1 to predictions...")
        preds = np.expm1(preds)
    # -------------------------
    
    # 8. Submit
    save_submission(
        ids=test_ids, 
        preds=preds, 
        id_col=config['id_column'], 
        target_col=config['target_column'],
        filename="submission.csv"
    )

if __name__ == "__main__":
    main()
