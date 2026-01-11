import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.config import ProjectConfig
from src.modeling.architecture import InterpretableMLP, SimpleCNN, RealCreditModel
from src.dataset import MNISTDataModule
from src.plots import VisualizationEngine
from src.data.credit_score_data import load_credit_score_dataset

class ModelTrainer:
    def __init__(self, model: nn.Module, config: ProjectConfig, criterion=None, lr=None) -> None:
        self.model = model
        self.config = config
        self.device = getattr(config, 'DEVICE', 'cpu')
        self.model.to(self.device)
        
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        learning_rate = lr if lr else config.LEARNING_RATE
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(self, train_loader, use_l1_regularization: bool) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            if not isinstance(self.model, SimpleCNN):
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            if use_l1_regularization:
                if hasattr(self.model, 'fc1'):
                    l1_norm = self.model.fc1.weight.abs().sum()
                    loss += self.config.L1_LAMBDA * l1_norm

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def save_model(self, filename: str) -> None:
        self.config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = self.config.MODELS_DIR / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

def train_real_credit_model(
    X: torch.Tensor, 
    y: torch.Tensor, 
    input_dim: int, 
    device: str, 
    epochs: int = 30, 
    batch_size: int = 256
) -> nn.Module:
    model = RealCreditModel(input_dim).to(device)
    
    class_counts = torch.bincount(y)
    weights = 1. / class_counts.float()
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

def run_mnist_experiment(config: ProjectConfig, viz: VisualizationEngine) -> None:
    data_module = MNISTDataModule(config)
    train_loader, _ = data_module.get_data_loaders()

    print(">>> Training Standard Model...")
    std_model = InterpretableMLP(
        config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE
    )
    std_trainer = ModelTrainer(std_model, config)
    std_losses = []

    for epoch in range(config.EPOCHS):
        loss = std_trainer.train_epoch(train_loader, use_l1_regularization=False)
        std_losses.append(loss)
        print(f"Standard Epoch {epoch+1}: Loss {loss:.4f}")

    std_trainer.save_model("standard_mlp.pth")
    viz.plot_training_curves(std_losses, "Standard Model Loss", "loss_standard.png")

    print("\n>>> Training Sparse Model (L1 Regularized)...")
    sparse_model = InterpretableMLP(
        config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE
    )
    sparse_trainer = ModelTrainer(sparse_model, config)
    sparse_losses = []

    for epoch in range(config.EPOCHS):
        loss = sparse_trainer.train_epoch(train_loader, use_l1_regularization=True)
        sparse_losses.append(loss)
        print(f"Sparse Epoch {epoch+1}: Loss {loss:.4f}")

    sparse_trainer.save_model("sparse_mlp.pth")
    viz.plot_training_curves(sparse_losses, "Sparse Model Loss", "loss_sparse.png")

    print("\n>>> Training Simple CNN...")
    cnn_model = SimpleCNN() 
    cnn_trainer = ModelTrainer(cnn_model, config)
    cnn_losses = []

    for epoch in range(config.EPOCHS):
        loss = cnn_trainer.train_epoch(train_loader, use_l1_regularization=False)
        cnn_losses.append(loss)
        print(f"CNN Epoch {epoch+1}: Loss {loss:.4f}")

    cnn_trainer.save_model("simple_cnn.pth")
    viz.plot_training_curves(cnn_losses, "CNN Model Loss", "loss_cnn.png")

def run_credit_scoring_experiment(config: ProjectConfig) -> None:
    print("\n>>> Training Real Credit Score Model...")
    try:
        df, X, y, _, _ = load_credit_score_dataset(max_samples=10000)
        device = getattr(config, 'DEVICE', 'cpu')
        
        model = train_real_credit_model(
            X, y, X.shape[1], device, epochs=30, batch_size=256
        )
        
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.MODELS_DIR / "real_credit_model.pth"
        torch.save(model.state_dict(), path)
        print(f"Credit Model saved to {path}")
        
    except FileNotFoundError:
        print("Skipping Credit Experiment: Dataset not found in data/raw/CreditScore/")

if __name__ == "__main__":
    conf = ProjectConfig()
    visualizer = VisualizationEngine(conf.FIGURES_DIR)
    
    run_mnist_experiment(conf, visualizer)
    run_credit_scoring_experiment(conf)