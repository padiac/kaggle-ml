import os
import pandas as pd

def save_submission(ids, preds, id_col, target_col, filename="submission.csv"):
    """
    Save predictions to a submission file.
    
    Args:
        ids: Series or array of IDs.
        preds: Series or array of predictions.
        id_col (str): specific ID column name required by competition.
        target_col (str): specific Target column name required by competition.
        filename (str): Output filename.
    """
    
    # Ensure numpy arrays are flattened if needed
    if hasattr(preds, 'flatten'):
        preds = preds.flatten()
        
    submission = pd.DataFrame({
        id_col: ids,
        target_col: preds
    })
    
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(submission.head())
