import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        
        self.layer1 = nn.Linear(5, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

def train_model(df: pd.DataFrame, epochs: int = 100, lr: float = 0.01, device: str = 'cpu') -> SimpleMLP:
    
    X = df[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']].values
    y = df['Target'].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    
    model = SimpleMLP().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    return model

def get_activations(model: SimpleMLP, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    
    h1 = model.layer1.register_forward_hook(get_activation('layer1'))
    h2 = model.layer2.register_forward_hook(get_activation('layer2'))
    
    
    model.eval()
    with torch.no_grad():
        model(input_tensor)
        
    
    h1.remove()
    h2.remove()
    
    return activations

def get_gradients(model: SimpleMLP, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of the output w.r.t. layer 1 and layer 2 activations.
    This is for 'Gradient-based Attribution'.
    """
    model.eval()
    
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
            output.retain_grad()
        return hook

    h1 = model.layer1.register_forward_hook(get_activation('layer1'))
    h2 = model.layer2.register_forward_hook(get_activation('layer2'))
    
    
    output = model(input_tensor)
    
    
    model.zero_grad()
    output.backward()
    
    
    grad1 = activations['layer1'].grad
    grad2 = activations['layer2'].grad
    
    h1.remove()
    h2.remove()
    
    return grad1, grad2

if __name__ == "__main__":
    
    from data_gen import generate_synthetic_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    df = generate_synthetic_data()
    model = train_model(df, epochs=50, device=device)
    print("Training complete.")
    
    
    sample = df.iloc[0][['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']].values
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    acts = get_activations(model, sample_tensor)
    print("Activations Layer 1 shape:", acts['layer1'].shape)
    
    
    g1, g2 = get_gradients(model, sample_tensor)
    print("Gradients Layer 1 shape:", g1.shape)
