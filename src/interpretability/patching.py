"""Activation Patching for causal circuit discovery.

Activation Patching (also known as causal tracing) is a technique for
establishing causal relationships between internal activations and model outputs.
By swapping activations between a "clean" run and a "corrupted" run, we can
identify which components are causally responsible for specific behaviors.

Reference: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

from src.interpretability.hooks import HookManager, ActivationCache


@dataclass
class PatchingResult:
    """Container for activation patching results."""

    clean_output: torch.Tensor
    corrupted_output: torch.Tensor
    patched_output: torch.Tensor
    layer_name: str
    effect: float  

    @property
    def recovery_ratio(self) -> float:
        """Compute how much of the corrupted->clean gap was recovered by patching.

        Returns:
            Value in [0, 1] where 1 means full recovery of clean behavior.
        """
        clean_corrupted_diff = (self.clean_output - self.corrupted_output).abs().mean()
        patched_corrupted_diff = (self.patched_output - self.corrupted_output).abs().mean()

        if clean_corrupted_diff < 1e-8:
            return 1.0

        return (patched_corrupted_diff / clean_corrupted_diff).item()


def activation_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    layer_name: str,
    neuron_indices: Optional[List[int]] = None,
) -> PatchingResult:
    """Perform activation patching from clean run to corrupted run.

    This is the core operation for causal tracing. We:
    1. Run the model on clean input, capturing activations at the target layer
    2. Run the model on corrupted input, but patch in the clean activations
    3. Measure how much the patched output resembles the clean output

    The intuition: if patching a layer's activations restores clean behavior,
    that layer is causally responsible for the difference in behavior.

    Args:
        model: The PyTorch model to analyze.
        clean_input: The "clean" input tensor (e.g., original prompt).
        corrupted_input: The "corrupted" input tensor (e.g., with key info changed).
        layer_name: The layer at which to perform patching.
        neuron_indices: Optional list of specific neurons to patch. If None, patch all.

    Returns:
        PatchingResult containing clean, corrupted, and patched outputs with metrics.
    """
    manager = HookManager(model)

    
    model.eval()
    with torch.no_grad():
        with manager.capture_activations([layer_name]) as cache:
            clean_output = model(clean_input)
        clean_activations = cache.get(layer_name)

    
    with torch.no_grad():
        corrupted_output = model(corrupted_input)

    
    def patch_fn(output: torch.Tensor) -> torch.Tensor:
        if neuron_indices is None:
            return clean_activations
        else:
            patched = output.clone()
            for idx in neuron_indices:
                if patched.dim() == 2:
                    patched[:, idx] = clean_activations[:, idx]
                else:
                    patched[idx] = clean_activations[idx]
            return patched

    with torch.no_grad():
        with manager.intervention(layer_name, patch_fn):
            patched_output = model(corrupted_input)

    
    clean_corrupted_dist = (clean_output - corrupted_output).abs().mean()
    patched_corrupted_dist = (patched_output - corrupted_output).abs().mean()

    if clean_corrupted_dist < 1e-8:
        effect = 0.0
    else:
        effect = (patched_corrupted_dist / clean_corrupted_dist).item()

    return PatchingResult(
        clean_output=clean_output,
        corrupted_output=corrupted_output,
        patched_output=patched_output,
        layer_name=layer_name,
        effect=effect,
    )


def causal_trace(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, PatchingResult]:
    """Perform causal tracing across all layers to find responsible components.

    This function runs activation patching on each layer and returns a map
    showing which layers, when patched, most strongly restore clean behavior.
    High-effect layers are causally important for the behavior change.

    Args:
        model: The PyTorch model.
        clean_input: The clean input tensor.
        corrupted_input: The corrupted input tensor.
        layer_names: Optional list of layers to trace. If None, traces all.

    Returns:
        Dictionary mapping layer names to their PatchingResult.
    """
    manager = HookManager(model)

    if layer_names is None:
        layer_names = manager.list_layers()

    results = {}
    for layer_name in layer_names:
        try:
            result = activation_patching(model, clean_input, corrupted_input, layer_name)
            results[layer_name] = result
        except ValueError:
            
            continue

    return results


def find_causal_layers(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    threshold: float = 0.5,
) -> List[Tuple[str, float]]:
    """Find layers that are causally responsible for behavior difference.

    Args:
        model: The PyTorch model.
        clean_input: The clean input tensor.
        corrupted_input: The corrupted input tensor.
        threshold: Minimum recovery ratio to consider a layer causal.

    Returns:
        List of (layer_name, effect) tuples sorted by effect descending.
    """
    trace_results = causal_trace(model, clean_input, corrupted_input)

    causal_layers = [
        (name, result.effect)
        for name, result in trace_results.items()
        if result.recovery_ratio >= threshold
    ]

    return sorted(causal_layers, key=lambda x: x[1], reverse=True)


def neuron_level_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    layer_name: str,
    num_neurons: int,
) -> Dict[int, float]:
    """Perform patching on individual neurons to find the most causal ones.

    Args:
        model: The PyTorch model.
        clean_input: The clean input tensor.
        corrupted_input: The corrupted input tensor.
        layer_name: The layer to analyze.
        num_neurons: Number of neurons in the layer.

    Returns:
        Dictionary mapping neuron index to its patching effect.
    """
    effects = {}

    for neuron_idx in range(num_neurons):
        result = activation_patching(
            model, clean_input, corrupted_input, layer_name, [neuron_idx]
        )
        effects[neuron_idx] = result.effect

    return effects


def steering_vector(
    model: nn.Module,
    positive_inputs: List[torch.Tensor],
    negative_inputs: List[torch.Tensor],
    layer_name: str,
) -> torch.Tensor:
    """Compute a steering vector from contrastive examples.

    The steering vector represents the direction in activation space that
    corresponds to moving from "negative" to "positive" behavior.
    This can be added to activations to steer model behavior without retraining.

    Args:
        model: The PyTorch model.
        positive_inputs: List of inputs exhibiting desired behavior.
        negative_inputs: List of inputs exhibiting undesired behavior.
        layer_name: The layer to compute the steering vector for.

    Returns:
        Steering vector that can be added to layer activations.
    """
    manager = HookManager(model)

    
    positive_activations = []
    model.eval()
    with torch.no_grad():
        for inp in positive_inputs:
            with manager.capture_activations([layer_name]) as cache:
                model(inp)
            positive_activations.append(cache.get(layer_name))

    
    negative_activations = []
    with torch.no_grad():
        for inp in negative_inputs:
            with manager.capture_activations([layer_name]) as cache:
                model(inp)
            negative_activations.append(cache.get(layer_name))

    
    pos_mean = torch.stack(positive_activations).mean(dim=0)
    neg_mean = torch.stack(negative_activations).mean(dim=0)

    return pos_mean - neg_mean


def apply_steering(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    steering_vec: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply a steering vector to modify model behavior.

    Args:
        model: The PyTorch model.
        input_tensor: The input to process.
        layer_name: The layer to apply steering at.
        steering_vec: The steering vector to add.
        strength: Multiplier for the steering vector.

    Returns:
        Model output with steering applied.
    """
    manager = HookManager(model)

    def steer_fn(output: torch.Tensor) -> torch.Tensor:
        return output + strength * steering_vec

    model.eval()
    with torch.no_grad():
        with manager.intervention(layer_name, steer_fn):
            output = model(input_tensor)

    return output
