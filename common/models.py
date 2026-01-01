from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_model(task, model_name, params=None):
    """
    Factory to retrieve a model instance.
    
    Args:
        task (str): 'classification' or 'regression'.
        model_name (str): Name of the model type.
        params (dict, optional): Dictionary of hyperparameters to override defaults.
        
    Returns:
        sklearn estimator
    """
    if params is None:
        params = {}
        
    model = None
    
    if task == 'classification':
        if model_name == 'logistic_regression':
            # Default options could be set here if not in params
            model = LogisticRegression(**params)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_name == 'gbm':
            model = GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
            
    elif task == 'regression':
        if model_name == 'ridge':
            model = Ridge(**params)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(**params)
        elif model_name == 'gbm':
            model = GradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
    else:
        raise ValueError(f"Unknown task: {task}")
        
    return model
