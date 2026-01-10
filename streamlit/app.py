"""Neural Microscope - Mechanistic Interpretability Dashboard.

This is the "View" layer that uses the generic interpretability module
from src.interpretability for analysis. The UI supports multiple demo modes:
- Credit Scoring (synthetic bias demonstration)
- MNIST (real trained models)
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import local modules
from data_gen import generate_synthetic_data
from model_utils import train_model, SimpleMLP

# Import generic interpretability tools from src
from src.interpretability import (
    HookManager,
    run_ablation,
    discover_circuits,
    find_top_k_neurons,
    prune_model,
)
from src.interpretability.hooks import get_all_activations
from src.interpretability.patching import activation_patching, causal_trace
from src.interpretability.pruning import prune_biased_neurons, evaluate_pruning_impact

from src.streamlit_tabs.activations import activations_tab
from src.streamlit_tabs.circut_detection import (
    circut_detection_tab,
    safety_pruning,
    circut_detection_on_images_tab,
)
from src.streamlit_tabs.ablation import ablation_tab, ablation_on_images_tab

st.set_page_config(page_title="Neural Microscope", layout="wide")
st.title("Neural Microscope: Mechanistic Interpretability Tool")
st.sidebar.header("Configuration")

demo_mode = st.sidebar.selectbox(
    "Demo Mode",
    ["Credit Scoring (Bias Demo)", "MNIST (Trained Models)"],
    index=0,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {device.upper()}")


# ============================================================================
# Credit Scoring Demo
# ============================================================================
if demo_mode == "Credit Scoring (Bias Demo)":
    st.markdown(
        """
        This dashboard visualizes how a neural network processes "Zip Code" (a proxy for redlining)
        to make credit decisions. We use the **src.interpretability** module to identify the specific
        neurons responsible for discrimination.
        """
    )

    n_samples = st.sidebar.slider("Dataset Size", 100, 2000, 1000)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)

    if st.sidebar.button("Regenerate Data"):
        st.cache_data.clear()
        st.cache_resource.clear()

    @st.cache_data
    def load_data(n: int, s: int):
        return generate_synthetic_data(n_samples=n, seed=s)

    @st.cache_resource
    def load_model_cached(_df, dev):
        return train_model(_df, epochs=200, device=dev)

    df = load_data(n_samples, seed)
    model = load_model_cached(df, device)

    st.sidebar.header("Applicant Selection")
    sample_idx = st.sidebar.slider("Select Applicant Index", 0, len(df) - 1, 0)

    sample = df.iloc[sample_idx]
    input_features = sample[
        ["Income", "Credit History", "Age", "Zip Code", "Random Noise"]
    ]
    target = sample["Target"]

    input_tensor = (
        torch.tensor(input_features.values, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # Forward Pass
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()

    # Display Applicant Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Applicant Profile")
        st.dataframe(input_features)
    with col2:
        st.subheader("Model Decision")
        st.metric("Credit Score (Probability)", f"{prediction:.4f}")
        st.metric("Actual Label", int(target))
    with col3:
        st.subheader("Bias Factor")
        zip_val = int(input_features["Zip Code"])
        st.write(
            f"Zip Code: {zip_val} ({'Advantaged' if zip_val == 1 else 'Disadvantaged'})"
        )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Activations", "Bias Circuit Detection", "Ablation Study", "Safety Pruning"]
    )

    manager = HookManager(model)
    layer_names = [name for name in manager.list_layers() if "layer" in name]
    activations = get_all_activations(model, input_tensor, layer_names)

    # ========== Tab 1: Activations ==========
    with tab1:
        activations_tab(model, input_tensor, activations, layer_names)

    # ========== Tab 2: Bias Circuit Detection ==========
    with tab2:
        circut_detection_tab(model, input_tensor, input_features, layer_names)

    # ========== Tab 3: Ablation Study ==========
    with tab3:
        ablation_tab(model, df, activations, layer_names)

    # ========== Tab 4: Safety Pruning ==========
    with tab4:
        safety_pruning(model, input_features, df)

# ============================================================================
# MNIST Demo
# ============================================================================
elif demo_mode == "MNIST (Trained Models)":
    st.header("MNIST Model Analysis")
    st.markdown(
        """
    Analyze trained MLP models on the MNIST dataset using the generic
    interpretability tools from `src.interpretability`.
    """
    )

    model_options = {
        "Standard MLP": "models/standard_mlp.pth",
        "Sparse MLP": "models/sparse_mlp.pth",
    }

    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    model_path = project_root / model_options[selected_model]

    @st.cache_resource
    def load_mnist_model(path: str, is_sparse: bool):
        from src.modeling.architecture import InterpretableMLP

        # Both models use the same architecture
        model = InterpretableMLP(input_size=784, hidden_size=128, output_size=10)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model

    @st.cache_data
    def load_mnist_data(split):
        train_loader, test_loader = load_mnist()
        if split == "train":
            return train_loader
        else:
            return test_loader

    try:
        is_sparse = "Sparse" in selected_model
        model = load_mnist_model(str(model_path), is_sparse)

        st.success(f"Loaded {selected_model}")

        st.subheader("Model Architecture")
        manager = HookManager(model)
        layers = manager.list_layers()

        layers_desc = f"`{layers[0]}`: {type(manager.get_layer(layers[0])).__name__}"
        for layer in layers[1:]:
            module = manager.get_layer(layer)
            layers_desc += f"  ->  `{layer}`: {type(module).__name__}"

        st.write(layers_desc)

        # Load sample MNIST data
        st.subheader("Sample Analysis")

        from src.dataset import load_mnist

        @st.cache_data
        def get_mnist_samples(n: int = 10):
            _, test_loader = load_mnist()
            images, labels = next(iter(test_loader))
            return images[:n], labels[:n]

        images, labels = get_mnist_samples(10)

        sample_idx = st.slider("Select sample", 0, 9, 0)
        sample_image = images[sample_idx : sample_idx + 1].view(1, -1)
        sample_label = labels[sample_idx].item()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**True Label**: {sample_label}")
            fig = px.imshow(
                images[sample_idx].squeeze().numpy(),
                color_continuous_scale="gray",
                title=f"Sample {sample_idx}",
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            with torch.no_grad():
                output = model(sample_image)
                pred = output.argmax(dim=1).item()
                probs = torch.softmax(output, dim=1).squeeze()

            st.write(f"**Predicted**: {pred}")

            fig = px.bar(
                x=list(range(10)),
                y=probs.numpy(),
                labels={"x": "Class", "y": "Probability"},
                title="Class Probabilities",
            )
            st.plotly_chart(fig, width="stretch")

        # # Layer activations
        # st.subheader("Layer Activations")

        # # Filter to main layers (Linear layers in Sequential)
        # analysis_layers = [l for l in layers if "network" in l]

        # if analysis_layers:
        #     selected_layer = st.selectbox("Select layer", analysis_layers)

        #     activations = get_all_activations(model, sample_image, [selected_layer])

        #     if selected_layer in activations:
        #         acts = activations[selected_layer].flatten()
        #         fig = go.Figure(
        #             data=go.Bar(
        #                 x=list(range(len(acts))), y=acts.numpy(), name=selected_layer
        #             )
        #         )
        #         fig.update_layout(
        #             xaxis_title="Neuron Index",
        #             yaxis_title="Activation",
        #             title=f"Activations in {selected_layer}",
        #         )
        #         st.plotly_chart(fig, width="stretch")

        #         # Top neurons
        #         top_neurons = find_top_k_neurons(
        #             model, sample_image, selected_layer, k=10, method="activation"
        #         )
        #         st.write("**Top 10 Active Neurons:**")
        #         for n in top_neurons:
        #             st.write(f"  Neuron {n.neuron_index}: {n.activation_mean:.4f}")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Activations",
                "Bias Circuit Detection",
                "Ablation Study",
                "Safety Pruning",
            ]
        )

        manager = HookManager(model)
        layer_names = [
            name for name in manager.list_layers() if "layer" in name or "fc" in name
        ]
        activations = get_all_activations(model, sample_image, layer_names)

        # ========== Tab 1: Activations ==========
        with tab1:
            activations_tab(
                model, sample_image, activations, layer_names, layer_prefix="fc"
            )

        # ========== Tab 2: Bias Circuit Detection ==========
        with tab2:
            st.error(f"Not implemented yet.")
            # circut_detection_on_images_tab(
            #     model, sample_image, 2, layer_names, layer_prefix="fc"
            # )

        # ========== Tab 3: Ablation Study ==========
        with tab3:
            test_loader = load_mnist_data("test")
            ablation_on_images_tab(model, test_loader, activations, layer_names)
            # ablation_tab(model, df, activations, layer_names)

        # ========== Tab 4: Safety Pruning ==========
        with tab4:
            st.error(f"Not implemented yet.")
            # safety_pruning(model, sample_image, df)

    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.info("Please train the model first using the training scripts.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
