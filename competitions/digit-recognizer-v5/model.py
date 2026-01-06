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
