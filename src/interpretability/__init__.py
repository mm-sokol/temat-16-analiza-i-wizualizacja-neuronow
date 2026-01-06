"""Interpretability module for mechanistic analysis of neural networks.

This module provides generic tools for:
- Hook management for capturing activations
- Ablation studies (zeroing neurons)
- Activation patching (causal tracing)
- Circuit discovery (identifying significant neurons)
- Safety pruning (permanently removing biased neurons)
"""

from src.interpretability.hooks import HookManager
from src.interpretability.ablation import run_ablation, ablate_neurons
from src.interpretability.patching import activation_patching, causal_trace
from src.interpretability.circuits import (
    discover_circuits,
    compute_neuron_importance,
    find_top_k_neurons,
)
from src.interpretability.pruning import prune_model, create_neuron_mask

__all__ = [
    "HookManager",
    "run_ablation",
    "ablate_neurons",
    "activation_patching",
    "causal_trace",
    "discover_circuits",
    "compute_neuron_importance",
    "find_top_k_neurons",
    "prune_model",
    "create_neuron_mask",
]
