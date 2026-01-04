import torch
import torch.nn as nn
import torch.optim as optim
from config import ProjectConfig
from modeling.architecture import InterpretableMLP
from dataset import MNISTDataModule
from plots import VisualizationEngine


class ModelTrainer:
    """
    Manages the training lifecycle.
    Includes logic for the L1 regularization experiment.
    """

    def __init__(self, model: InterpretableMLP, config: ProjectConfig) -> None:
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    def train_epoch(self, train_loader, use_l1_regularization: bool) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            if use_l1_regularization:
                # EXPERIMENT: Enforce sparsity in the first layer weights
                # This simulates 'circuit discovery' by removing noisy connections
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


def run_training_experiment() -> None:
    """
    Trains two models for comparison:
    1. Standard Baseline
    2. Sparsified Model (L1 Regularized)
    """
    config = ProjectConfig()
    viz = VisualizationEngine(config.FIGURES_DIR)

    # Data
    data_module = MNISTDataModule(config)
    train_loader, _ = data_module.get_data_loaders()

    # 1. Train Standard Model
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

    # 2. Train Sparse Model (The Experiment)
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


if __name__ == "__main__":
    run_training_experiment()
