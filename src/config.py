from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    """
    Central configuration for the Neural Network Interpreter project.
    Contains paths, hyperparameters, and experiment settings.
    """

    
    ROOT_DIR: Path = Path(__file__).resolve().parents[1]

    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"

    MODELS_DIR: Path = ROOT_DIR / "models"
    FIGURES_DIR: Path = ROOT_DIR / "reports" / "figures"

    
    RANDOM_SEED: int = 42
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 20

    
    
    L1_LAMBDA: float = 0.0005

    
    INPUT_SIZE: int = 784  
    HIDDEN_SIZE: int = 128
    OUTPUT_SIZE: int = 10  

    INTERPRETER: str = "integrated_grad"
