"""Circuit discovery utilities for identifying significant neurons.

This module provides tools for automatically identifying which neurons
are most important for specific model behaviors, forming the basis
for understanding the "circuits" that implement model functionality.

A circuit is a subgraph of the network that implements a specific computation.
By identifying the neurons most responsible for a behavior, we can begin
to reverse-engineer the algorithm the model has learned.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.interpretability.hooks import HookManager, get_all_activations
from src.interpretability.ablation import compute_ablation_effects


@dataclass
class NeuronInfo:
    """Information about a significant neuron."""

    layer_name: str
    neuron_index: int
    importance_score: float
    activation_mean: float
    activation_std: float


@dataclass
class CircuitInfo:
    """Information about a discovered circuit."""

    neurons: List[NeuronInfo]
    total_importance: float
    description: str = ""

    def get_neuron_indices(self, layer_name: str) -> List[int]:
        """Get indices of neurons belonging to a specific layer."""
        return [n.neuron_index for n in self.neurons if n.layer_name == layer_name]

    def get_mask(self, layer_name: str, num_neurons: int) -> torch.Tensor:
        """Create a binary mask for neurons in this circuit."""
        mask = torch.zeros(num_neurons, dtype=torch.bool)
        for idx in self.get_neuron_indices(layer_name):
            mask[idx] = True
        return mask


def compute_neuron_importance(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    method: str = "gradient",
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute importance scores for neurons in a layer.

    Importance quantifies how much each neuron contributes to the output.
    Multiple methods are supported:
    - "gradient": Gradient of output w.r.t. neuron activations (fast)
    - "ablation": Change in output when neuron is zeroed (causal but slow)
    - "activation": Raw activation magnitude (simplest)

    Args:
        model: The PyTorch model.
        input_tensor: The input to analyze.
        layer_name: The layer to compute importance for.
        method: The importance computation method.
        target_index: Optional output index to focus on.

    Returns:
        Tensor of importance scores for each neuron.
    """
    if method == "activation":
        return _importance_by_activation(model, input_tensor, layer_name)
    elif method == "gradient":
        return _importance_by_gradient(model, input_tensor, layer_name, target_index)
    elif method == "ablation":
        return _importance_by_ablation(model, input_tensor, layer_name, target_index)
    else:
        raise ValueError(f"Unknown importance method: {method}")


def _importance_by_activation(
    model: nn.Module, input_tensor: torch.Tensor, layer_name: str
) -> torch.Tensor:
    """Compute importance as activation magnitude."""
    activations = get_all_activations(model, input_tensor, [layer_name])
    act = activations[layer_name]

    # Take absolute mean across batch
    if act.dim() == 2:
        return act.abs().mean(dim=0)
    elif act.dim() == 1:
        return act.abs()
    else:
        # For higher-dimensional activations, flatten spatial dims
        return act.abs().mean(dim=tuple(range(2, act.dim()))).mean(dim=0)


def _importance_by_gradient(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute importance as gradient of output w.r.t. activations."""
    manager = HookManager(model)
    layer = manager.get_layer(layer_name)

    if layer is None:
        raise ValueError(f"Layer {layer_name} not found")

    # We need to capture activations with gradients
    captured_activation = None

    def capture_hook(module, input, output):
        nonlocal captured_activation
        captured_activation = output
        output.retain_grad()
        return output

    handle = layer.register_forward_hook(capture_hook)

    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)
    output = model(input_tensor)

    if target_index is not None:
        target = output[:, target_index].sum()
    else:
        target = output.sum()

    target.backward()

    handle.remove()

    if captured_activation.grad is not None:
        grad = captured_activation.grad.abs()
        if grad.dim() == 2:
            importance = grad.mean(dim=0)
        else:
            importance = grad.mean(dim=tuple(range(1, grad.dim())))
    else:
        # Fallback if gradient not captured
        importance = torch.ones(captured_activation.shape[-1])

    return importance.detach()


def _importance_by_ablation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute importance by ablation effect."""
    # Get layer size from activations
    activations = get_all_activations(model, input_tensor, [layer_name])
    act = activations[layer_name]

    if act.dim() >= 2:
        num_neurons = act.shape[1]
    else:
        num_neurons = act.shape[0]

    return compute_ablation_effects(
        model, input_tensor, layer_name, num_neurons, target_index
    ).abs()


def find_top_k_neurons(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    k: int = 10,
    method: str = "gradient",
    target_index: Optional[int] = None,
) -> List[NeuronInfo]:
    """Find the top-k most important neurons in a layer.

    Args:
        model: The PyTorch model.
        input_tensor: The input to analyze.
        layer_name: The layer to analyze.
        k: Number of top neurons to return.
        method: Importance computation method.
        target_index: Optional output index to focus on.

    Returns:
        List of NeuronInfo for the top-k neurons, sorted by importance.
    """
    importance = compute_neuron_importance(
        model, input_tensor, layer_name, method, target_index
    )

    # Get activation statistics
    activations = get_all_activations(model, input_tensor, [layer_name])
    act = activations[layer_name]
    if act.dim() == 2:
        act_mean = act.mean(dim=0)
        # Avoid warning when batch size is 1
        act_std = act.std(dim=0) if act.shape[0] > 1 else torch.zeros_like(act_mean)
    else:
        act_mean = act
        act_std = torch.zeros_like(act)

    # Get top-k indices
    k = min(k, len(importance))
    top_values, top_indices = torch.topk(importance, k)

    neurons = []
    for i, idx in enumerate(top_indices.tolist()):
        neurons.append(
            NeuronInfo(
                layer_name=layer_name,
                neuron_index=idx,
                importance_score=top_values[i].item(),
                activation_mean=act_mean[idx].item(),
                activation_std=act_std[idx].item() if act_std.numel() > 0 else 0.0,
            )
        )

    return neurons


def discover_circuits(
    model: nn.Module,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    layer_names: Optional[List[str]] = None,
    top_k: int = 5,
    method: str = "gradient",
) -> CircuitInfo:
    """Discover the circuit responsible for a prediction difference.

    This function identifies neurons across layers that are most responsible
    for the difference in predictions between two inputs. This helps locate
    the "circuit" implementing a specific behavior or discrimination.

    Args:
        model: The PyTorch model.
        input_a: First input tensor.
        input_b: Second input tensor (contrastive).
        layer_names: Layers to analyze. If None, analyzes all.
        top_k: Number of top neurons per layer to include.
        method: Importance computation method.

    Returns:
        CircuitInfo containing the identified circuit neurons.
    """
    manager = HookManager(model)

    if layer_names is None:
        layer_names = manager.list_layers()

    # Get predictions
    model.eval()
    with torch.no_grad():
        pred_a = model(input_a)
        pred_b = model(input_b)

    # Get target based on prediction difference
    pred_diff = (pred_a - pred_b).abs()
    if pred_diff.dim() > 1:
        target_index = pred_diff.squeeze().argmax().item()
    else:
        target_index = None

    # Collect important neurons from each layer
    all_neurons = []

    for layer_name in layer_names:
        try:
            # Compute importance on both inputs and look for differential importance
            importance_a = compute_neuron_importance(
                model, input_a, layer_name, method, target_index
            )
            importance_b = compute_neuron_importance(
                model, input_b, layer_name, method, target_index
            )

            # Neurons important for the difference
            diff_importance = (importance_a - importance_b).abs()

            # Get activation stats from input_a
            activations = get_all_activations(model, input_a, [layer_name])
            act = activations[layer_name]
            if act.dim() == 2:
                act_mean = act.mean(dim=0)
                # Avoid warning when batch size is 1
                act_std = act.std(dim=0) if act.shape[0] > 1 else torch.zeros_like(act_mean)
            else:
                act_mean = act
                act_std = torch.zeros_like(act)

            # Get top-k for this layer
            k = min(top_k, len(diff_importance))
            top_values, top_indices = torch.topk(diff_importance, k)

            for i, idx in enumerate(top_indices.tolist()):
                all_neurons.append(
                    NeuronInfo(
                        layer_name=layer_name,
                        neuron_index=idx,
                        importance_score=top_values[i].item(),
                        activation_mean=act_mean[idx].item(),
                        activation_std=act_std[idx].item() if act_std.numel() > 0 else 0.0,
                    )
                )
        except (ValueError, RuntimeError):
            # Skip layers that can't be analyzed
            continue

    # Sort all neurons by importance
    all_neurons.sort(key=lambda n: n.importance_score, reverse=True)

    total_importance = sum(n.importance_score for n in all_neurons)

    return CircuitInfo(
        neurons=all_neurons,
        total_importance=total_importance,
        description=f"Circuit for differentiating inputs at output {target_index}",
    )


def get_layer_importance_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    method: str = "gradient",
    target_index: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute importance map for all layers.

    Args:
        model: The PyTorch model.
        input_tensor: The input to analyze.
        method: Importance computation method.
        target_index: Optional output index to focus on.

    Returns:
        Dictionary mapping layer names to importance tensors.
    """
    manager = HookManager(model)
    importance_map = {}

    for layer_name in manager.list_layers():
        try:
            importance = compute_neuron_importance(
                model, input_tensor, layer_name, method, target_index
            )
            importance_map[layer_name] = importance
        except (ValueError, RuntimeError):
            continue

    return importance_map
