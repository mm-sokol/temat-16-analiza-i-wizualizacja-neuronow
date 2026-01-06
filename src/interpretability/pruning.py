"""Safety pruning utilities for permanently removing biased or harmful neurons.

This module provides methods for surgically removing neurons identified as
problematic (e.g., encoding protected attributes like zip code for redlining)
without fully retraining the model. This enables post-hoc bias mitigation.

Pruning permanently modifies model weights, unlike ablation which is temporary.
The trade-off between fairness and accuracy should be carefully monitored.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import dataclass

from src.interpretability.hooks import HookManager


@dataclass
class PruningResult:
    """Container for pruning operation results."""

    original_model: nn.Module
    pruned_model: nn.Module
    pruned_layers: Dict[str, List[int]]  # layer_name -> pruned neuron indices
    num_neurons_pruned: int

    def summary(self) -> str:
        """Generate a summary of the pruning operation."""
        lines = [
            f"Pruned {self.num_neurons_pruned} neurons across {len(self.pruned_layers)} layers:"
        ]
        for layer, indices in self.pruned_layers.items():
            lines.append(
                f"  {layer}: {len(indices)} neurons ({indices[:5]}{'...' if len(indices) > 5 else ''})"
            )
        return "\n".join(lines)


def create_neuron_mask(
    num_neurons: int,
    neurons_to_prune: List[int],
    invert: bool = False,
) -> torch.Tensor:
    """Create a binary mask for neuron pruning.

    Args:
        num_neurons: Total number of neurons in the layer.
        neurons_to_prune: List of neuron indices to prune.
        invert: If True, creates mask where pruned neurons are 1 (for selection).
                If False, creates mask where kept neurons are 1 (for masking).

    Returns:
        Boolean tensor of shape (num_neurons,).
    """
    mask = torch.ones(num_neurons, dtype=torch.bool)
    for idx in neurons_to_prune:
        if 0 <= idx < num_neurons:
            mask[idx] = False

    if invert:
        mask = ~mask

    return mask


def prune_model(
    model: nn.Module,
    neuron_mask: Dict[str, Union[torch.Tensor, List[int]]],
    copy: bool = True,
) -> nn.Module:
    """Permanently prune neurons by zeroing their incoming and outgoing weights.

    This function modifies the model weights to effectively remove specified neurons.
    For a linear layer y = Wx + b:
    - Zeroing column j of W removes the contribution of input feature j
    - Zeroing row i of W and b[i] removes neuron i's output

    For neurons to be truly removed, we zero both the row in the current layer
    and the corresponding column in the next layer (if applicable).

    Args:
        model: The PyTorch model to prune.
        neuron_mask: Dictionary mapping layer names to masks or lists of neurons to prune.
                     If a Tensor, True means keep, False means prune.
                     If a List, contains indices of neurons to prune.
        copy: If True, creates a deep copy and prunes the copy.

    Returns:
        The pruned model (copy if copy=True, otherwise in-place modification).
    """
    if copy:
        model = deepcopy(model)

    # Convert lists to masks
    processed_masks = {}
    for layer_name, mask_or_list in neuron_mask.items():
        if isinstance(mask_or_list, list):
            # Get layer to determine size
            manager = HookManager(model)
            layer = manager.get_layer(layer_name)
            if layer is None:
                continue

            # Determine number of neurons
            if hasattr(layer, "out_features"):
                num_neurons = layer.out_features
            elif hasattr(layer, "weight"):
                num_neurons = layer.weight.shape[0]
            else:
                continue

            processed_masks[layer_name] = create_neuron_mask(num_neurons, mask_or_list)
        else:
            processed_masks[layer_name] = mask_or_list

    # Apply pruning
    for layer_name, mask in processed_masks.items():
        _prune_layer(model, layer_name, mask)

    return model


def _prune_layer(model: nn.Module, layer_name: str, keep_mask: torch.Tensor) -> None:
    """Prune a single layer by zeroing weights of pruned neurons.

    Args:
        model: The model containing the layer.
        layer_name: Name of the layer to prune.
        keep_mask: Boolean mask where True = keep, False = prune.
    """
    manager = HookManager(model)
    layer = manager.get_layer(layer_name)

    if layer is None:
        return

    with torch.no_grad():
        if hasattr(layer, "weight"):
            # Zero out rows corresponding to pruned neurons
            prune_indices = (~keep_mask).nonzero(as_tuple=True)[0]
            for idx in prune_indices:
                if idx < layer.weight.shape[0]:
                    layer.weight.data[idx] = 0.0

                    if hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias.data[idx] = 0.0


def prune_by_importance(
    model: nn.Module,
    layer_name: str,
    importance_scores: torch.Tensor,
    threshold: Optional[float] = None,
    keep_ratio: Optional[float] = None,
    keep_top: Optional[int] = None,
    copy: bool = True,
) -> nn.Module:
    """Prune neurons based on importance scores.

    Neurons with low importance scores (below threshold or bottom percentile)
    are pruned. Exactly one of threshold, keep_ratio, or keep_top must be specified.

    Args:
        model: The model to prune.
        layer_name: The layer to prune.
        importance_scores: Importance score for each neuron.
        threshold: Absolute threshold; prune if importance < threshold.
        keep_ratio: Keep this fraction of neurons (e.g., 0.8 keeps top 80%).
        keep_top: Keep exactly this many neurons.
        copy: Create a copy before pruning.

    Returns:
        Pruned model.
    """
    num_neurons = len(importance_scores)

    if sum(x is not None for x in [threshold, keep_ratio, keep_top]) != 1:
        raise ValueError(
            "Exactly one of threshold, keep_ratio, or keep_top must be specified"
        )

    if threshold is not None:
        keep_mask = importance_scores >= threshold
    elif keep_ratio is not None:
        num_keep = int(num_neurons * keep_ratio)
        _, top_indices = torch.topk(importance_scores, num_keep)
        keep_mask = torch.zeros(num_neurons, dtype=torch.bool)
        keep_mask[top_indices] = True
    else:  # keep_top
        _, top_indices = torch.topk(importance_scores, min(keep_top, num_neurons))
        keep_mask = torch.zeros(num_neurons, dtype=torch.bool)
        keep_mask[top_indices] = True

    return prune_model(model, {layer_name: keep_mask}, copy=copy)


def prune_biased_neurons(
    model: nn.Module,
    layer_name: str,
    biased_neuron_indices: List[int],
    copy: bool = True,
) -> Tuple[nn.Module, PruningResult]:
    """Convenience function to prune neurons identified as encoding bias.

    This is the main entry point for safety-focused pruning. After identifying
    neurons that encode protected attributes (e.g., via circuit discovery),
    this function removes them from the model.

    Args:
        model: The model to prune.
        layer_name: The layer containing biased neurons.
        biased_neuron_indices: List of neuron indices to prune.
        copy: Create a copy before pruning.

    Returns:
        Tuple of (pruned_model, PruningResult with details).
    """
    pruned = prune_model(model, {layer_name: biased_neuron_indices}, copy=copy)

    result = PruningResult(
        original_model=model if copy else None,
        pruned_model=pruned,
        pruned_layers={layer_name: biased_neuron_indices},
        num_neurons_pruned=len(biased_neuron_indices),
    )

    return pruned, result


def evaluate_pruning_impact(
    original_model: nn.Module,
    pruned_model: nn.Module,
    test_inputs: torch.Tensor,
    test_labels: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Evaluate the impact of pruning on model performance.

    Args:
        original_model: The original unpruned model.
        pruned_model: The pruned model.
        test_inputs: Test input tensors.
        test_labels: Optional ground truth labels for accuracy computation.

    Returns:
        Dictionary with metrics comparing original and pruned performance.
    """
    original_model.eval()
    pruned_model.eval()

    with torch.no_grad():
        original_outputs = original_model(test_inputs)
        pruned_outputs = pruned_model(test_inputs)

    # Compute output difference
    output_diff = (original_outputs - pruned_outputs).abs()

    metrics = {
        "mean_output_change": output_diff.mean().item(),
        "max_output_change": output_diff.max().item(),
        "output_correlation": torch.corrcoef(
            torch.stack([original_outputs.flatten(), pruned_outputs.flatten()])
        )[0, 1].item()
        if original_outputs.numel() > 1
        else 1.0,
    }

    # Compute accuracy if labels provided
    if test_labels is not None:
        if original_outputs.dim() > 1 and original_outputs.shape[-1] > 1:
            original_preds = original_outputs.argmax(dim=-1)
            pruned_preds = pruned_outputs.argmax(dim=-1)
        else:
            original_preds = (original_outputs > 0.5).squeeze().long()
            pruned_preds = (pruned_outputs > 0.5).squeeze().long()

        metrics["original_accuracy"] = (
            (original_preds == test_labels).float().mean().item()
        )
        metrics["pruned_accuracy"] = (pruned_preds == test_labels).float().mean().item()
        metrics["accuracy_drop"] = (
            metrics["original_accuracy"] - metrics["pruned_accuracy"]
        )
        metrics["prediction_agreement"] = (
            (original_preds == pruned_preds).float().mean().item()
        )

    return metrics


def iterative_pruning(
    model: nn.Module,
    layer_name: str,
    importance_fn,
    target_sparsity: float,
    steps: int = 5,
    copy: bool = True,
) -> nn.Module:
    """Gradually prune neurons over multiple steps.

    Iterative pruning can be gentler than one-shot pruning, as the model
    can partially adapt between steps (if fine-tuning is interleaved).

    Args:
        model: The model to prune.
        layer_name: The layer to prune.
        importance_fn: Function that takes model and returns importance scores.
        target_sparsity: Final fraction of neurons to prune (0.3 = prune 30%).
        steps: Number of pruning iterations.
        copy: Create a copy before pruning.

    Returns:
        Pruned model.
    """
    if copy:
        model = deepcopy(model)

    # Calculate neurons to prune per step
    importance = importance_fn(model)
    num_neurons = len(importance)
    total_to_prune = int(num_neurons * target_sparsity)
    per_step = total_to_prune // steps

    pruned_so_far = set()

    for step in range(steps):
        # Recompute importance
        importance = importance_fn(model)

        # Mask out already pruned neurons
        for idx in pruned_so_far:
            importance[idx] = float("inf")

        # Find next batch to prune
        if step == steps - 1:
            # Last step: prune remaining
            num_to_prune = total_to_prune - len(pruned_so_far)
        else:
            num_to_prune = per_step

        if num_to_prune <= 0:
            continue

        # Get least important neurons
        _, bottom_indices = torch.topk(importance, num_to_prune, largest=False)
        new_prune = set(bottom_indices.tolist())
        pruned_so_far.update(new_prune)

        # Apply pruning
        model = prune_model(model, {layer_name: list(new_prune)}, copy=False)

    return model
