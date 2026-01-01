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
        
        # Layer 1: Input to Hidden (Feature extraction)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Layer 2: Hidden to Output (Classification)
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