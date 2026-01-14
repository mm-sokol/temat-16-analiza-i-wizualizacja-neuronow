import torch
import torch.nn as nn

class InterpretableMLP(nn.Module):
    """
    Multi-Layer Perceptron designed for mechanistic interpretability.
    
    Structure:
    Input (784) -> Linear (FC1) -> ReLU -> Linear (FC2) -> Output (10)
    
    We use a simple structure to easily visualize weight matrices and
    identify active circuits.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 10) 

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
    


class RealCreditModel(nn.Module):
    """
    Model for real-world credit scoring data (Kaggle).
    Includes Batch Normalization and Dropout for regularization.
    """
    def __init__(self, input_dim: int, hidden_dim: list = [64, 32], output_dim: int = 3) -> None:
        super().__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])
        self.relu2 = nn.ReLU()
        
        
        self.output = nn.Linear(hidden_dim[1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu2(self.bn2(self.layer2(x)))
        return self.output(x)