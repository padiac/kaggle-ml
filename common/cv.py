import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

def run_cv(model_pipeline, X, y, task, n_splits=5, random_state=42, scoring=None):
    """
    Run cross-validation and print/return results.
    
    Args:
        model_pipeline: The scikit-learn pipeline/model.
        X: Feature matrix.
        y: Target vector.
        task (str): 'classification' or 'regression'.
        n_splits (int): Number of folds.
        random_state (int): Random seed.
        scoring (str): Scikit-learn scoring metric.
        
    Returns:
        scores (list): List of CV scores.
    """
    
    if task == 'classification':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif task == 'regression':
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown task: {task}")
        
    print(f"Running {n_splits}-fold CV ({task}) with scoring='{scoring}'...")
    
    scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring=scoring)
    
    print(f"CV Scores: {scores}")
    print(f"CV Score: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    
    return scores
