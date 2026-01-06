"""Ablation utilities for zeroing out specific neurons in neural networks.

Ablation is a core technique in mechanistic interpretability for understanding
the causal role of individual neurons or groups of neurons. By setting specific
activations to zero and observing the change in model output, we can infer
the functional importance of those neurons.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from src.interpretability.hooks import HookManager


def ablate_neurons(
    output: torch.Tensor,
    neuron_indices: List[int],
    ablation_value: float = 0.0,
) -> torch.Tensor:
    """Zero out (or set to a specific value) neurons at given indices.

    This is the core ablation operation that modifies a tensor in-place by
    setting specified neuron activations to a constant value.

    Args:
        output: The activation tensor to modify (batch_size, num_neurons, ...).
        neuron_indices: List of neuron indices to ablate.
        ablation_value: Value to set ablated neurons to (default: 0.0).

    Returns:
        Modified tensor with specified neurons ablated.
    """
    output_modified = output.clone()
    for idx in neuron_indices:
        if output_modified.dim() == 1:
            output_modified[idx] = ablation_value
        elif output_modified.dim() == 2:
            output_modified[:, idx] = ablation_value
        elif output_modified.dim() >= 3:
            # For conv layers or higher-dimensional outputs
            output_modified[:, idx, ...] = ablation_value
    return output_modified


def run_ablation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    neuron_indices: List[int],
    ablation_value: float = 0.0,
) -> torch.Tensor:
    """Run a forward pass with specific neurons ablated (zeroed out).

    This function performs a single forward pass through the model while
    intercepting and modifying the activations at a specified layer.
    The mathematical effect is: h'_i = 0 for i in neuron_indices.

    Args:
        model: The PyTorch model to run ablation on.
        input_tensor: The input tensor to pass through the model.
        layer_name: The name of the layer to ablate (as returned by named_modules).
        neuron_indices: List of neuron indices within the layer to ablate.
        ablation_value: Value to set ablated neurons to (default: 0.0).

    Returns:
        The model output with the specified neurons ablated.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        >>> x = torch.randn(1, 10)
        >>> output = run_ablation(model, x, "0", [0, 2])  # Ablate neurons 0 and 2
    """
    manager = HookManager(model)

    def intervention_fn(output: torch.Tensor) -> torch.Tensor:
        return ablate_neurons(output, neuron_indices, ablation_value)

    model.eval()
    with torch.no_grad():
        with manager.intervention(layer_name, intervention_fn):
            output = model(input_tensor)

    return output


def batch_ablation_study(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    num_neurons: int,
) -> Dict[int, torch.Tensor]:
    """Perform ablation study on each neuron individually.

    This function runs the model once for each neuron in the specified layer,
    ablating only that neuron, to measure the individual contribution of each.

    Args:
        model: The PyTorch model.
        input_tensor: The input tensor.
        layer_name: The layer to study.
        num_neurons: Number of neurons in the layer.

    Returns:
        Dictionary mapping neuron index to the model output when that neuron is ablated.
    """
    results = {}

    for neuron_idx in range(num_neurons):
        ablated_output = run_ablation(model, input_tensor, layer_name, [neuron_idx])
        results[neuron_idx] = ablated_output

    return results


def compute_ablation_effects(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    num_neurons: int,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute the effect of ablating each neuron on the output.

    For each neuron, computes: effect_i = output_original - output_ablated_i
    This quantifies how much each neuron contributes to the final output.

    Args:
        model: The PyTorch model.
        input_tensor: The input tensor.
        layer_name: The layer to study.
        num_neurons: Number of neurons in the layer.
        target_index: If provided, only compute effect on this output index.

    Returns:
        Tensor of shape (num_neurons,) containing the effect of ablating each neuron.
    """
    model.eval()
    with torch.no_grad():
        original_output = model(input_tensor)

    ablation_results = batch_ablation_study(model, input_tensor, layer_name, num_neurons)

    effects = []
    for neuron_idx in range(num_neurons):
        ablated_output = ablation_results[neuron_idx]
        effect = original_output - ablated_output

        if target_index is not None:
            effect = effect[:, target_index]

        effects.append(effect.mean().item())

    return torch.tensor(effects)


def mean_ablation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    neuron_indices: List[int],
    reference_activations: torch.Tensor,
) -> torch.Tensor:
    """Ablate neurons by replacing with mean activation from reference dataset.

    Instead of zeroing, this replaces activations with their mean value
    computed over a reference dataset, which can be a gentler intervention.

    Args:
        model: The PyTorch model.
        input_tensor: The input tensor.
        layer_name: The layer to ablate.
        neuron_indices: Neurons to ablate.
        reference_activations: Mean activations from reference dataset.

    Returns:
        Model output with mean ablation applied.
    """
    manager = HookManager(model)

    def intervention_fn(output: torch.Tensor) -> torch.Tensor:
        output_modified = output.clone()
        for idx in neuron_indices:
            if output_modified.dim() == 2:
                output_modified[:, idx] = reference_activations[idx]
            else:
                output_modified[idx] = reference_activations[idx]
        return output_modified

    model.eval()
    with torch.no_grad():
        with manager.intervention(layer_name, intervention_fn):
            output = model(input_tensor)

    return output
