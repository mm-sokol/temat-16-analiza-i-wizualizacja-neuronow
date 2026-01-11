import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod

# Próbujemy zaimportować Captum. Jeśli go nie ma, kod zadziała, ale bez tych metod.
try:
    from captum.attr import Saliency, IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class ModelInterpreter(ABC):

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @abstractmethod
    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        pass

    def get_layer_weights(self, layer_name: str) -> torch.Tensor:
        """
        Retrieves weight matrix from a specific linear layer for structural visualization.
        """
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                return module.weight.detach()

        raise ValueError(f"Layer '{layer_name}' not found or is not a Linear layer.")


class GradientInterpreter(ModelInterpreter):
    """
    Component responsible for mechanistic interpretability analysis.
    Extracts internal model states and computes feature importance (saliency).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        """
        Computes the gradient of the target class score with respect to the input image.
        This answers: 'Which pixels caused the model to choose this class?'
        """
        self.model.eval()

        # We need gradients relative to the input image
        input_tensor.requires_grad_()

        # Forward pass
        output = self.model(input_tensor)

        # Zero previous gradients
        self.model.zero_grad()

        # Backward pass for the specific target class score
        target_score = output[0, target_class]
        target_score.backward()

        # Saliency is the magnitude of the gradient (how sensitive is the output to this pixel?)
        if input_tensor.grad is not None:
            saliency = input_tensor.grad.data.abs()
            return saliency
        else:
            raise RuntimeError(
                "Gradient computation failed. Ensure input requires_grad."
            )


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
        return ig.abs()



class CaptumSaliency(ModelInterpreter):
    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum library is not installed.")
            
        algo = Saliency(self.model)
        attr = algo.attribute(input_tensor, target=target_class, abs=False)
        return attr.abs()


class CaptumIG(ModelInterpreter):
    def compute_saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, **kwargs
    ) -> torch.Tensor:
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum library is not installed.")

        algo = IntegratedGradients(self.model)
        attr = algo.attribute(input_tensor, target=target_class, n_steps=50)
        return attr.abs()



INTERPRETER_REGISTRY = {
    "gradient": GradientInterpreter,
    "integrated_grad": IntegratedGradient,
    "captum_saliency": CaptumSaliency,
    "captum_ig": CaptumIG,
}


def interpreter_factory(interpreter_type: str, model):
    cls = INTERPRETER_REGISTRY.get(interpreter_type)
    if cls is None:
        raise ValueError(f"Unimplemented interpreter type: {interpreter_type}")
    return cls(model)