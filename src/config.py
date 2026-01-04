from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    """
    Central configuration for the Neural Network Interpreter project.
    Contains paths, hyperparameters, and experiment settings.
    """

    # Paths - resolving relative to src/config.py
    ROOT_DIR: Path = Path(__file__).resolve().parents[1]

    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"

    MODELS_DIR: Path = ROOT_DIR / "models"
    FIGURES_DIR: Path = ROOT_DIR / "reports" / "figures"

    # Hyperparameters
    RANDOM_SEED: int = 42
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 20

    # Regularization Experiment
    # L1 penalty induces sparsity (dead neurons), acting as a proxy for interpretability
    L1_LAMBDA: float = 0.0005

    # Model Architecture (Simple MLP for MNIST)
    INPUT_SIZE: int = 784  # 28x28 pixels
    HIDDEN_SIZE: int = 128
    OUTPUT_SIZE: int = 10  # 10 digits

    INTERPRETER: str = "integrated_grad"
