import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from typing import List

class VisualizationEngine:
    """
    Handles generation and saving of interpretability plots.
    Adheres to orthogonality: logic is purely about visualization.
    """
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_saliency(self, original_image: torch.Tensor, saliency_map: torch.Tensor, filename: str) -> None:
        """
        Visualizes the original image side-by-side with its saliency map.
        """
        img_np = original_image.squeeze().cpu().detach().numpy()
        sal_np = saliency_map.squeeze().cpu().detach().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title("Original Input")
        ax1.axis('off')
        
        # Saliency Heatmap
        ax2.imshow(img_np, cmap='gray', alpha=0.5)
        im = ax2.imshow(sal_np, cmap='hot', alpha=0.9)
        ax2.set_title("Saliency Map (Decision Focus)")
        ax2.axis('off')
        
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_weight_matrix(self, weights: torch.Tensor, title: str, filename: str) -> None:
        """
        Visualizes the weight matrix of a layer. 
        Crucial for the regularization experiment: shows sparsity (dead neurons).
        """
        w_np = weights.cpu().detach().numpy()
        
        plt.figure(figsize=(12, 8))
        # vmin/vmax centered around 0 to show positive/negative weights clearly
        plt.imshow(w_np, aspect='auto', cmap='seismic', vmin=-0.5, vmax=0.5)
        plt.colorbar(label="Weight Strength")
        plt.title(f"{title} - Structural Visualization")
        plt.xlabel("Input Connections (Pixels)")
        plt.ylabel("Hidden Neurons")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_training_curves(self, losses: List[float], title: str, filename: str) -> None:
        """
        Plots the loss trajectory over epochs.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(losses, marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
