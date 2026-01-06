"""Generic hook management for PyTorch neural networks.

Provides a context-manager-based approach to register and manage forward hooks
on any named layer of a torch.nn.Module, enabling activation capture and
intervention during forward passes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class ActivationCache:
    """Container for storing captured activations from hooks."""

    activations: Dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all stored activations."""
        self.activations.clear()

    def get(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get activation for a specific layer."""
        return self.activations.get(layer_name)

    def items(self):
        """Iterate over layer names and activations."""
        return self.activations.items()

    def keys(self):
        """Get all layer names with cached activations."""
        return self.activations.keys()


class HookManager:
    """Manages forward hooks on PyTorch modules for activation capture and intervention.

    This class provides a clean interface for registering hooks on any named layer
    of a neural network. It supports both activation capture (for analysis) and
    activation modification (for interventions like ablation or patching).

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        >>> manager = HookManager(model)
        >>> with manager.capture_activations(['0', '2']) as cache:
        ...     output = model(input_tensor)
        ...     print(cache.activations)
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the hook manager with a model.

        Args:
            model: The PyTorch model to attach hooks to.
        """
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._layer_map: Dict[str, nn.Module] = {}
        self._build_layer_map()

    def _build_layer_map(self) -> None:
        """Build a mapping from layer names to layer modules."""
        for name, module in self.model.named_modules():
            if name:  # Skip the root module
                self._layer_map[name] = module

    def get_layer(self, name: str) -> Optional[nn.Module]:
        """Get a layer by name.

        Args:
            name: The name of the layer (as returned by named_modules).

        Returns:
            The layer module if found, None otherwise.
        """
        return self._layer_map.get(name)

    def list_layers(self) -> List[str]:
        """List all available layer names.

        Returns:
            List of layer names that can be hooked.
        """
        return list(self._layer_map.keys())

    def register_hook(
        self,
        layer_name: str,
        hook_fn: Callable[[nn.Module, Any, torch.Tensor], Optional[torch.Tensor]],
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on a specific layer.

        Args:
            layer_name: The name of the layer to hook.
            hook_fn: The hook function with signature (module, input, output) -> Optional[Tensor].
                     If it returns a Tensor, that becomes the new output.

        Returns:
            A handle that can be used to remove the hook.

        Raises:
            ValueError: If the layer name is not found.
        """
        layer = self.get_layer(layer_name)
        if layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found. Available layers: {self.list_layers()}"
            )

        handle = layer.register_forward_hook(hook_fn)
        self._handles.append(handle)
        return handle

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def capture_activations(
        self, layer_names: List[str], detach: bool = True
    ) -> "ActivationCaptureContext":
        """Context manager for capturing activations from specified layers.

        Args:
            layer_names: List of layer names to capture activations from.
            detach: Whether to detach captured tensors from the computation graph.

        Returns:
            A context manager that yields an ActivationCache.
        """
        return ActivationCaptureContext(self, layer_names, detach)

    def intervention(
        self,
        layer_name: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> "InterventionContext":
        """Context manager for applying an intervention to a layer's output.

        Args:
            layer_name: The layer to intervene on.
            intervention_fn: Function that takes the layer output and returns modified output.

        Returns:
            A context manager that applies the intervention during forward passes.
        """
        return InterventionContext(self, layer_name, intervention_fn)


class ActivationCaptureContext:
    """Context manager for capturing activations during forward passes."""

    def __init__(
        self, manager: HookManager, layer_names: List[str], detach: bool = True
    ) -> None:
        self.manager = manager
        self.layer_names = layer_names
        self.detach = detach
        self.cache = ActivationCache()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> ActivationCache:
        self.cache.clear()

        for layer_name in self.layer_names:

            def make_hook(name: str):
                def hook(module: nn.Module, input: Any, output: torch.Tensor):
                    if self.detach:
                        self.cache.activations[name] = output.detach().clone()
                    else:
                        self.cache.activations[name] = output.clone()

                return hook

            handle = self.manager.register_hook(layer_name, make_hook(layer_name))
            self._handles.append(handle)

        return self.cache

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
            if handle in self.manager._handles:
                self.manager._handles.remove(handle)


class InterventionContext:
    """Context manager for applying interventions to layer outputs."""

    def __init__(
        self,
        manager: HookManager,
        layer_name: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.manager = manager
        self.layer_name = layer_name
        self.intervention_fn = intervention_fn
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def __enter__(self) -> "InterventionContext":
        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
            return self.intervention_fn(output)

        self._handle = self.manager.register_hook(self.layer_name, hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._handle:
            self._handle.remove()
            if self._handle in self.manager._handles:
                self.manager._handles.remove(self._handle)


def get_all_activations(
    model: nn.Module, input_tensor: torch.Tensor, layer_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """Convenience function to get activations from a model in a single forward pass.

    Args:
        model: The PyTorch model.
        input_tensor: The input tensor to pass through the model.
        layer_names: Optional list of layer names to capture. If None, captures all.

    Returns:
        Dictionary mapping layer names to their activation tensors.
    """
    manager = HookManager(model)

    if layer_names is None:
        layer_names = manager.list_layers()

    model.eval()
    with torch.no_grad():
        with manager.capture_activations(layer_names) as cache:
            model(input_tensor)

    return dict(cache.activations)
