from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class SklearnCNN(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=10, batch_size=64, learning_rate=0.001, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.loss_curve_ = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for CNN models.")
            
        torch.manual_seed(self.random_state)
        
        # Prepare Data
        # X comes in as (N, 784), need to reshape to (N, 1, 28, 28)
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).view(-1, 1, 28, 28).to(self.device)
        y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y).to(self.device)
        
        # Define Model (Simple CNN)
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 classes
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.loss_curve_ = []
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Store average loss per batch or total loss
            self.loss_curve_.append(epoch_loss / len(dataloader))
            
        return self
        
    def predict(self, X):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for CNN models.")
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).view(-1, 1, 28, 28).to(self.device)
        
        # Predict in batches to avoid OOM
        predictions = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(X_tensor, batch_size=self.batch_size)
            for batch_X in dataloader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                
        return np.array(predictions)

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
    
    # Extract meta-parameters that are not meant for the model constructor
    multiclass_strategy = params.pop('multiclass_strategy', None)
        
    model = None
    
    if task == 'classification':
        if model_name == 'logistic_regression':
            # Use MLP with no hidden layers to simulate Logistic Regression
            # This allows us to track loss_curve_
            # Convert LR params to MLP params where possible/needed
            mlp_params = {
                'hidden_layer_sizes': (), # No hidden layers = Linear
                'activation': 'identity', # Not strictly needed for output usually, but good for explicit
                'solver': 'adam', # Force 'adam' (or 'sgd') because 'lbfgs' does not support loss_curve_
                'early_stopping': True, # Enable validation scoring
                'validation_fraction': 0.1,
                'max_iter': params.get('max_iter', 200),
                'random_state': params.get('random_state', 42)
            }
            # Remove sklearn LR specific params that crash MLP
            # e.g. 'l1_ratio', 'C' (MLP uses alpha)
            
            model = MLPClassifier(**mlp_params)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_name == 'gbm':
            model = GradientBoostingClassifier(**params)
        elif model_name == 'mlp':
            model = MLPClassifier(**params)
        elif model_name == 'cnn':
            model = SklearnCNN(**params)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
            
    elif task == 'regression':
        if model_name == 'ridge':
            model = Ridge(**params)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(**params)
        elif model_name == 'gbm':
            model = GradientBoostingRegressor(**params)
        elif model_name == 'mlp':
            model = MLPRegressor(**params)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
    else:
        raise ValueError(f"Unknown task: {task}")
        
    
    # Handle Multiclass Strategy (e.g. One-vs-Rest)
    if multiclass_strategy == 'ovr':
        from sklearn.multiclass import OneVsRestClassifier
        print(f"Wrapping {model_name} in OneVsRestClassifier")
        model = OneVsRestClassifier(model)
            
    return model
