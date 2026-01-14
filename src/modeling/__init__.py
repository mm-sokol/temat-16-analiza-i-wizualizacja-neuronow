"""Modeling module for neural network architectures and training."""

from src.modeling.architecture import (
    InterpretableMLP,
    SimpleCNN,
    RealCreditModel,
)
from src.modeling.train import (
    ModelTrainer,
    train_real_credit_model,
    run_mnist_experiment,
    run_credit_scoring_experiment,
)

__all__ = [
    # Architectures
    "InterpretableMLP",
    "SimpleCNN",
    "RealCreditModel",
    # Training
    "ModelTrainer",
    "train_real_credit_model",
    "run_mnist_experiment",
    "run_credit_scoring_experiment",
]
