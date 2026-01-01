import sys
import os
import yaml
import pandas as pd
import numpy as np

# Add project root to path to import common modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from common.data import load_data
from common.preprocessing import get_preprocessor
from common.models import get_model
from common.cv import run_cv
from common.submit import save_submission
from sklearn.pipeline import Pipeline

def main():
    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Starting {config['competition_name']} - {config['task']}")
    
    # 1. Load Data
    train_df, test_df = load_data(config['competition_name'])
    
    # 2. Feature Engineering
    print("Feature Engineering: creating FamilySize and IsAlone...")
    train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
    test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
    train_df["IsAlone"] = (train_df["FamilySize"] == 1).astype(int)
    test_df["IsAlone"]  = (test_df["FamilySize"] == 1).astype(int)

    # 3. Clean Columns
    # Keep IDs for submission
    train_ids = train_df[config['id_column']]
    test_ids = test_df[config['id_column']]
    
    # Drop columns as requested
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    # Only drop if they exist to avoid errors if not present
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors='ignore')

    # Prepare X and y
    # Note: target_column might have been dropped above if it was in drop_cols (unlikely for Survived)
    y = train_df[config['target_column']]
    X = train_df.drop(columns=[config['target_column']])
    X_test = test_df.copy()

    # 4. Select Features based on Config
    # The config defines what we pull into the pipeline
    numeric_cols = config['features']['numeric']
    categorical_cols = config['features']['categorical']
    
    print(f"Using Numeric: {numeric_cols}")
    print(f"Using Categorical: {categorical_cols}")
    
    X = X[numeric_cols + categorical_cols]
    X_test = X_test[numeric_cols + categorical_cols]
    
    # 5. Preprocessing Pipeline
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    
    # 6. Model
    model_factory = get_model(config['task'], config['model']['name'], config['model'].get('params', {}))
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_factory)
    ])
    
    # 7. Cross Validation
    scores = run_cv(
        pipeline, 
        X, 
        y, 
        task=config['task'], 
        n_splits=config['cv']['n_splits'], 
        scoring=config['cv']['scoring']
    )
    
    # 8. Train on Full Data
    print("Training on full dataset...")
    pipeline.fit(X, y)
    
    # 9. Train Accuracy Check
    train_pred = pipeline.predict(X)
    train_acc = (train_pred == y).mean()
    print(f"Train accuracy: {train_acc:.5f}")
    
    # 10. Predict
    print("Predicting on test set...")
    preds = pipeline.predict(X_test)
    
    # 11. Submit
    save_submission(
        ids=test_ids, 
        preds=preds, 
        id_col=config['id_column'], 
        target_col=config['target_column'],
        filename="submission.csv"
    )

if __name__ == "__main__":
    main()
