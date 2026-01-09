import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

# Próba importu Captum - jeśli nie ma, kod nie wybuchnie od razu
try:
    from captum.attr import Saliency, IntegratedGradients, DeepLift, LayerConductance
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: Captum not installed. Captum interpreters will fail.")


class ModelInterpreter(ABC):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        pass

    def get_layer_weights(self, layer_name: str) -> torch.Tensor:
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                return module.weight.detach()

        raise ValueError(f"Layer '{layer_name}' not found or is not a Linear layer.")


# --- WASZE IMPLEMENTACJE (Manualne) ---

class GradientInterpreter(ModelInterpreter):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        self.model.eval()
        input_tensor.requires_grad_()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        target_score.backward()

        if input_tensor.grad is not None:
            return input_tensor.grad.data.abs()
        else:
            raise RuntimeError("Gradient computation failed.")


class IntegratedGradient(ModelInterpreter):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, steps=64, **kwargs
    ) -> torch.Tensor:
        baseline_tensor = torch.zeros_like(input_tensor)
        alphas = torch.linspace(0, 1, steps)
        gradients = []

        for alpha in alphas:
            interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor)
            interpolated.requires_grad_(True)
            
            output = self.model(interpolated)
            target_score = output[0, target_class]
            self.model.zero_grad()
            target_score.backward()
            
            gradients.append(interpolated.grad.detach())

        integral_approx = (
            gradients[0] / 2
            + torch.mean(torch.stack(gradients[1:-1]), dim=0)
            + gradients[-1] / 2
        )
        ig = (input_tensor - baseline_tensor) * integral_approx
        return ig.abs().max(dim=1)[0]


# --- IMPLEMENTACJE CAPTUM ---

class CaptumSaliency(ModelInterpreter):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.algorithm = Saliency(model) if CAPTUM_AVAILABLE else None

    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        if not self.algorithm: raise ImportError("Captum not available")
        input_tensor.requires_grad_()
        return self.algorithm.attribute(input_tensor, target=target_class, abs_mask=True)


class CaptumIntegratedGradients(ModelInterpreter):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.algorithm = IntegratedGradients(model) if CAPTUM_AVAILABLE else None

    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, n_steps=50, **kwargs
    ) -> torch.Tensor:
        if not self.algorithm: raise ImportError("Captum not available")
        attr = self.algorithm.attribute(
            input_tensor, 
            target=target_class, 
            n_steps=n_steps,
            baselines=torch.zeros_like(input_tensor)
        )
        return torch.abs(attr)


class CaptumDeepLift(ModelInterpreter):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.algorithm = DeepLift(model) if CAPTUM_AVAILABLE else None

    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        if not self.algorithm: raise ImportError("Captum not available")
        attr = self.algorithm.attribute(input_tensor, target=target_class)
        return torch.abs(attr)


# --- REJESTR ---

INTERPRETER_REGISTRY = {
    # Domyślne/Stare (Wasz kod)
    "gradient": GradientInterpreter,
    "integrated_grad": IntegratedGradient,

    # Nowe (Captum)
    "captum_gradient": CaptumSaliency,
    "captum_ig": CaptumIntegratedGradients,
    "captum_deeplift": CaptumDeepLift,
}


def interpreter_factory(interpreter_type: str, model: nn.Module):
    cls = INTERPRETER_REGISTRY.get(interpreter_type)
    if cls is None:
        valid_keys = list(INTERPRETER_REGISTRY.keys())
        raise ValueError(f"Interpreter '{interpreter_type}' not found. Valid options: {valid_keys}")
    return cls(model)