import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from typing import List, Union
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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

def visualize_activations(activations: Union[torch.Tensor, np.ndarray], layer_name: str):
    """
    Renders interactive activation maps or bar charts in Streamlit.
    Automatically detects if the input is from a CNN (3D) or MLP (1D).
    
    Args:
        activations: The tensor/array containing activation values.
        layer_name: Name of the layer for labeling purposes.
    """
    # 1. Safe conversion to NumPy
    if isinstance(activations, torch.Tensor):
        acts = activations.detach().cpu().numpy()
    else:
        acts = activations
        
    # 2. Squeeze batch dimension if present (e.g., [1, 128] -> [128])
    if acts.ndim > 1 and acts.shape[0] == 1:
        acts = acts.squeeze(0)

    # --- SCENARIO A: CNN (3D -> [Channels, Height, Width]) ---
    if acts.ndim == 3:
        n_channels = acts.shape[0]
        st.markdown(f"#### 🧠 Convolutional Layer: `{layer_name}`")
        st.caption(f"Detected {n_channels} feature maps. Each map responds to a specific visual pattern.")
        
        # Channel Selector
        col1, col2 = st.columns([1, 3])
        with col1:
            channel_idx = st.slider(f"Select Channel", 0, n_channels-1, 0, key=f"slider_{layer_name}")
        
        with col2:
            fig = px.imshow(
                acts[channel_idx],
                color_continuous_scale='Viridis',
                title=f"Activation Map: Channel {channel_idx}",
                labels={'x': 'W', 'y': 'H', 'color': 'Activation'},
                height=400
            )
            # Clean up axes
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- SCENARIO B: MLP (1D -> [Neurons]) ---
    elif acts.ndim == 1:
        n_neurons = len(acts)
        st.markdown(f"#### ⚡ Linear Layer: `{layer_name}`")
        
        # Use Histogram for very large layers to avoid clutter
        if n_neurons > 500:
            st.warning("High neuron count. Displaying activation distribution histogram.")
            fig = px.histogram(
                acts, 
                nbins=50, 
                title="Activation Distribution",
                labels={'value': 'Activation Value'}
            )
        # Use Bar Chart for standard layers
        else:
            fig = px.bar(
                x=list(range(n_neurons)),
                y=acts,
                title="Neuron Activations",
                labels={'x': 'Neuron Index', 'y': 'Activation Value'},
                color=acts,
                color_continuous_scale='Bluered'
            )
            fig.update_layout(showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # --- SCENARIO C: Unsupported Shape ---
    else:
        st.error(f"⚠️ Unsupported activation shape: {acts.shape}. Expected 1D (MLP) or 3D (CNN).")