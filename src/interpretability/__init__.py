"""Interpretability module for mechanistic analysis of neural networks.

This module provides generic tools for:
- Hook management for capturing activations
- Ablation studies (zeroing neurons)
- Activation patching (causal tracing)
- Circuit discovery (identifying significant neurons)
- Safety pruning (permanently removing biased neurons)
"""

from src.interpretability.hooks import HookManager, get_all_activations, ActivationCache
from src.interpretability.ablation import run_ablation, ablate_neurons
from src.interpretability.patching import activation_patching, causal_trace
from src.interpretability.circuits import (
    discover_circuits,
    compute_neuron_importance,
    find_top_k_neurons,
)
from src.interpretability.pruning import (
    prune_model,
    create_neuron_mask,
    prune_biased_neurons,
    evaluate_pruning_impact,
    PruningResult,
)

__all__ = [
    # Hooks
    "HookManager",
    "get_all_activations",
    "ActivationCache",
    # Ablation
    "run_ablation",
    "ablate_neurons",
    # Patching
    "activation_patching",
    "causal_trace",
    # Circuits
    "discover_circuits",
    "compute_neuron_importance",
    "find_top_k_neurons",
    # Pruning
    "prune_model",
    "create_neuron_mask",
    "prune_biased_neurons",
    "evaluate_pruning_impact",
    "PruningResult",
]
