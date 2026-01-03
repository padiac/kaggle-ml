from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_preprocessor(numeric_features, categorical_features):
    """
    Creates a ColumnTransformer for preprocessing.
    
    Args:
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.
        
    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
        
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # or 'passthrough' if you want other columns
    )
    
    return preprocessor
