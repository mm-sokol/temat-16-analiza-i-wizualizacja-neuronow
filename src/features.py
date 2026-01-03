import torch
import torch.nn as nn
from typing import Optional

class ModelInterpreter:
    """
    Component responsible for mechanistic interpretability analysis.
    Extracts internal model states and computes feature importance (saliency).
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def compute_saliency_map(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
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
            raise RuntimeError("Gradient computation failed. Ensure input requires_grad.")

    def get_layer_weights(self, layer_name: str) -> torch.Tensor:
        """
        Retrieves weight matrix from a specific linear layer for structural visualization.
        """
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                return module.weight.detach()
        
        raise ValueError(f"Layer '{layer_name}' not found or is not a Linear layer.")
