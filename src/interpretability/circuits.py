import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.interpretability.hooks import HookManager, get_all_activations
from src.interpretability.ablation import compute_ablation_effects


@dataclass
class NeuronInfo:
    layer_name: str
    neuron_index: int
    importance_score: float
    activation_mean: float
    activation_std: float


@dataclass
class CircuitInfo:
    neurons: List[NeuronInfo]
    total_importance: float
    description: str = ""

    def get_neuron_indices(self, layer_name: str) -> List[int]:
        return [n.neuron_index for n in self.neurons if n.layer_name == layer_name]

    def get_mask(self, layer_name: str, num_neurons: int) -> torch.Tensor:
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
    activations = get_all_activations(model, input_tensor, [layer_name])
    act = activations[layer_name]

    if act.dim() == 2:
        return act.abs().mean(dim=0)
    elif act.dim() == 1:
        return act.abs()
    else:
        return act.abs().mean(dim=tuple(range(2, act.dim()))).mean(dim=0)


def _importance_by_gradient(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    manager = HookManager(model)
    layer = manager.get_layer(layer_name)

    if layer is None:
        raise ValueError(f"Layer {layer_name} not found")

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
        importance = torch.ones(captured_activation.shape[-1])

    return importance.detach()


def _importance_by_ablation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    target_index: Optional[int] = None,
) -> torch.Tensor:
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
    importance = compute_neuron_importance(
        model, input_tensor, layer_name, method, target_index
    )

    activations = get_all_activations(model, input_tensor, [layer_name])
    act = activations[layer_name]
    if act.dim() == 2:
        act_mean = act.mean(dim=0)
        act_std = act.std(dim=0) if act.shape[0] > 1 else torch.zeros_like(act_mean)
    else:
        act_mean = act
        act_std = torch.zeros_like(act)

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
    manager = HookManager(model)

    if layer_names is None:
        layer_names = manager.list_layers()

    model.eval()
    with torch.no_grad():
        pred_a = model(input_a)
        pred_b = model(input_b)

    pred_diff = (pred_a - pred_b).abs()
    
    if pred_diff.dim() > 1 and pred_diff.shape[-1] > 1:
        target_index = pred_diff.sum(dim=0).argmax().item()
    else:
        target_index = 0

    all_neurons = []

    for layer_name in layer_names:
        try:
            importance_a = compute_neuron_importance(
                model, input_a, layer_name, method, target_index
            )
            importance_b = compute_neuron_importance(
                model, input_b, layer_name, method, target_index
            )

            diff_importance = (importance_a - importance_b).abs()

            activations = get_all_activations(model, input_a, [layer_name])
            act = activations[layer_name]
            if act.dim() == 2:
                act_mean = act.mean(dim=0)
                act_std = act.std(dim=0) if act.shape[0] > 1 else torch.zeros_like(act_mean)
            else:
                act_mean = act
                act_std = torch.zeros_like(act)

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
            continue

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