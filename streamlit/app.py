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

st.set_page_config(page_title="Neural Microscope", layout="wide")
st.title("Neural Microscope: Mechanistic Interpretability Tool")

# Sidebar configuration
st.sidebar.header("Configuration")

demo_mode = st.sidebar.selectbox(
    "Demo Mode",
    ["Credit Scoring (Bias Demo)", "MNIST (Trained Models)"],
    index=0,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.write(f"Device: {device.upper()}")


# ============================================================================
# Credit Scoring Demo
# ============================================================================
if demo_mode == "Credit Scoring (Bias Demo)":
    st.markdown("""
    This dashboard visualizes how a neural network processes "Zip Code" (a proxy for redlining)
    to make credit decisions. We use the **src.interpretability** module to identify the specific
    neurons responsible for discrimination.
    """)

    # Sidebar controls
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

    # Sample selection
    st.sidebar.header("Applicant Selection")
    sample_idx = st.sidebar.slider("Select Applicant Index", 0, len(df) - 1, 0)

    # Get Sample
    sample = df.iloc[sample_idx]
    input_features = sample[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']]
    target = sample['Target']

    # Prepare Tensor
    input_tensor = torch.tensor(
        input_features.values, dtype=torch.float32
    ).unsqueeze(0).to(device)

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
        zip_val = int(input_features['Zip Code'])
        st.write(f"Zip Code: {zip_val} ({'Advantaged' if zip_val == 1 else 'Disadvantaged'})")

    st.divider()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Activations",
        "Bias Circuit Detection",
        "Ablation Study",
        "Safety Pruning"
    ])

    # ========== Tab 1: Activations ==========
    with tab1:
        st.header("Inside the Black Box")
        st.info("Using generic HookManager from src.interpretability")

        # Use generic HookManager
        manager = HookManager(model)
        layer_names = [name for name in manager.list_layers() if 'layer' in name]

        # Get activations using generic function
        activations = get_all_activations(model, input_tensor, layer_names)

        col_l1, col_l2 = st.columns(2)

        with col_l1:
            if 'layer1' in activations:
                l1_acts = activations['layer1'].cpu().numpy().flatten()
                st.subheader(f"Layer 1 Activations ({len(l1_acts)} Neurons)")
                fig_l1 = px.imshow(
                    [l1_acts],
                    labels=dict(x="Neuron Index", y="Layer", color="Activation"),
                    color_continuous_scale="Viridis",
                    height=200
                )
                fig_l1.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_l1, width='stretch')

        with col_l2:
            if 'layer2' in activations:
                l2_acts = activations['layer2'].cpu().numpy().flatten()
                st.subheader(f"Layer 2 Activations ({len(l2_acts)} Neurons)")
                fig_l2 = px.imshow(
                    [l2_acts],
                    labels=dict(x="Neuron Index", y="Layer", color="Activation"),
                    color_continuous_scale="Viridis",
                    height=200
                )
                fig_l2.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_l2, width='stretch')

        # Neuron importance analysis
        st.subheader("Top Neurons by Importance")
        importance_method = st.selectbox("Importance Method", ["activation", "gradient"])

        for layer_name in layer_names:
            try:
                top_neurons = find_top_k_neurons(
                    model, input_tensor, layer_name, k=5, method=importance_method
                )
                st.write(f"**{layer_name}** - Top 5 neurons:")
                for n in top_neurons:
                    st.write(
                        f"  Neuron {n.neuron_index}: "
                        f"importance={n.importance_score:.4f}, "
                        f"activation={n.activation_mean:.4f}"
                    )
            except Exception as e:
                st.warning(f"Could not analyze {layer_name}: {e}")

    # ========== Tab 2: Bias Circuit Detection ==========
    with tab2:
        st.header("Detecting the Discrimination Circuit")
        st.markdown("""
        **Experiment:** What if we change *only* the Zip Code for this applicant?
        We flip the Zip Code (0 -> 1 or 1 -> 0) and observe which neurons change
        their activation the most. These neurons form the **Bias Circuit**.
        """)

        # Create Counterfactual
        cf_features = input_features.copy()
        cf_features['Zip Code'] = 1 - cf_features['Zip Code']  # Flip
        cf_tensor = torch.tensor(
            cf_features.values, dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Get activations for both using generic module
        orig_activations = get_all_activations(model, input_tensor, layer_names)
        cf_activations = get_all_activations(model, cf_tensor, layer_names)

        # Calculate Differences
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            if 'layer1' in orig_activations:
                l1_acts = orig_activations['layer1'].cpu().numpy().flatten()
                cf_l1_acts = cf_activations['layer1'].cpu().numpy().flatten()
                diff_l1 = cf_l1_acts - l1_acts

                st.subheader("Layer 1 Sensitivity (Bias Neurons)")
                fig_d1 = px.bar(
                    x=list(range(len(diff_l1))),
                    y=diff_l1,
                    labels={'x': 'Neuron Index', 'y': 'Change in Activation'},
                    title="Change when Zip Code is Flipped"
                )
                st.plotly_chart(fig_d1, width='stretch')

        with col_d2:
            if 'layer2' in orig_activations:
                l2_acts = orig_activations['layer2'].cpu().numpy().flatten()
                cf_l2_acts = cf_activations['layer2'].cpu().numpy().flatten()
                diff_l2 = cf_l2_acts - l2_acts

                st.subheader("Layer 2 Sensitivity (Bias Neurons)")
                fig_d2 = px.bar(
                    x=list(range(len(diff_l2))),
                    y=diff_l2,
                    labels={'x': 'Neuron Index', 'y': 'Change in Activation'},
                    title="Change when Zip Code is Flipped"
                )
                st.plotly_chart(fig_d2, width='stretch')

        st.info(
            "Neurons with large bars are strongly coupled to the Zip Code feature. "
            "These are candidates for pruning to improve fairness."
        )

        # Circuit discovery using generic module
        st.subheader("Automated Circuit Discovery")
        if st.button("Discover Bias Circuit"):
            circuit = discover_circuits(
                model, input_tensor, cf_tensor, layer_names, top_k=5
            )

            st.write(f"**Total Circuit Importance:** {circuit.total_importance:.4f}")
            st.write("**Identified Neurons:**")

            for neuron in circuit.neurons[:10]:
                st.write(
                    f"- **{neuron.layer_name}** Neuron {neuron.neuron_index}: "
                    f"importance={neuron.importance_score:.4f}"
                )

            # Store in session state for pruning tab
            st.session_state['bias_circuit'] = circuit
            st.success("Circuit saved! Go to 'Safety Pruning' tab to apply interventions.")

    # ========== Tab 3: Ablation Study ==========
    with tab3:
        st.header("Ablation Study")
        st.info("Using run_ablation from src.interpretability.ablation")

        selected_layer = st.selectbox("Select layer to ablate", layer_names, key="ablation_layer")

        if selected_layer and selected_layer in activations:
            n_neurons = activations[selected_layer].flatten().shape[0]

            neurons_to_ablate = st.multiselect(
                f"Select neurons to ablate (0-{n_neurons - 1})",
                list(range(n_neurons)),
                default=[],
                key="ablation_neurons"
            )

            if neurons_to_ablate and st.button("Run Ablation Study"):
                # Compare predictions across dataset
                original_probs = []
                ablated_probs = []

                progress = st.progress(0)
                for idx, row in df.iterrows():
                    x = torch.tensor(
                        row[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']].values,
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    # Original prediction
                    with torch.no_grad():
                        orig_prob = model(x).item()

                    # Ablated prediction using generic module
                    ablated_output = run_ablation(model, x, selected_layer, neurons_to_ablate)
                    abl_prob = ablated_output.item()

                    original_probs.append(orig_prob)
                    ablated_probs.append(abl_prob)

                    progress.progress((idx + 1) / len(df))

                comparison_df = df.copy()
                comparison_df["Original Prob"] = original_probs
                comparison_df["Ablated Prob"] = ablated_probs
                comparison_df["Difference"] = comparison_df["Ablated Prob"] - comparison_df["Original Prob"]

                st.subheader("Ablation Results")

                # Scatter plot
                fig = px.scatter(
                    comparison_df,
                    x="Original Prob",
                    y="Ablated Prob",
                    color="Zip Code",
                    hover_data=["Income", "Credit History"],
                    title="Original vs Ablated Predictions"
                )
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="No Change",
                    line=dict(dash="dash")
                ))
                st.plotly_chart(fig, width='stretch')

                # Average effect by zip code
                st.subheader("Average Effect by Zip Code")
                avg_diff_by_zip = comparison_df.groupby("Zip Code")["Difference"].mean()
                st.bar_chart(avg_diff_by_zip)

    # ========== Tab 4: Safety Pruning ==========
    with tab4:
        st.header("Safety Pruning")
        st.info("Permanently remove biased neurons using src.interpretability.pruning")

        st.markdown("""
        **Goal:** Remove neurons that encode protected attributes (Zip Code)
        to improve model fairness while minimizing accuracy loss.
        """)

        # Check if circuit was discovered
        if 'bias_circuit' not in st.session_state:
            st.warning("No bias circuit discovered yet. Go to 'Bias Circuit Detection' tab first.")
        else:
            circuit = st.session_state['bias_circuit']

            # Group neurons by layer
            layer_neurons = {}
            for n in circuit.neurons:
                if n.layer_name not in layer_neurons:
                    layer_neurons[n.layer_name] = []
                layer_neurons[n.layer_name].append((n.neuron_index, n.importance_score))

            st.subheader("Discovered Bias Neurons")
            for layer, neurons in layer_neurons.items():
                st.write(f"**{layer}:** {[n[0] for n in neurons]}")

            # Pruning controls
            prune_layer = st.selectbox(
                "Layer to prune",
                list(layer_neurons.keys()),
                key="prune_layer"
            )

            if prune_layer:
                available_neurons = [n[0] for n in layer_neurons[prune_layer]]
                prune_neurons = st.multiselect(
                    "Neurons to prune",
                    available_neurons,
                    default=available_neurons[:2] if len(available_neurons) >= 2 else available_neurons,
                    key="prune_neurons"
                )

                if prune_neurons and st.button("Apply Pruning"):
                    # Prune the model
                    pruned_model, result = prune_biased_neurons(
                        model, prune_layer, prune_neurons, copy=True
                    )

                    st.write(result.summary())

                    # Compare on contrastive examples
                    st.subheader("Fairness Impact")

                    # Create matched pairs with different zip codes
                    test_features_adv = input_features.copy()
                    test_features_adv['Zip Code'] = 1
                    test_features_dis = input_features.copy()
                    test_features_dis['Zip Code'] = 0

                    x_adv = torch.tensor(
                        test_features_adv.values, dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    x_dis = torch.tensor(
                        test_features_dis.values, dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        orig_adv = model(x_adv).item()
                        orig_dis = model(x_dis).item()
                        pruned_adv = pruned_model(x_adv).item()
                        pruned_dis = pruned_model(x_dis).item()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Advantaged Zip",
                            f"{pruned_adv:.3f}",
                            delta=f"{pruned_adv - orig_adv:.3f}"
                        )
                    with col2:
                        st.metric(
                            "Disadvantaged Zip",
                            f"{pruned_dis:.3f}",
                            delta=f"{pruned_dis - orig_dis:.3f}"
                        )
                    with col3:
                        old_gap = orig_adv - orig_dis
                        new_gap = pruned_adv - pruned_dis
                        st.metric(
                            "Bias Gap",
                            f"{new_gap:.3f}",
                            delta=f"{new_gap - old_gap:.3f}",
                            delta_color="inverse"
                        )

                    # Full dataset comparison
                    st.subheader("Full Dataset Impact")

                    orig_probs = []
                    pruned_probs = []

                    for idx, row in df.iterrows():
                        x = torch.tensor(
                            row[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']].values,
                            dtype=torch.float32
                        ).unsqueeze(0).to(device)

                        with torch.no_grad():
                            orig_probs.append(model(x).item())
                            pruned_probs.append(pruned_model(x).item())

                    comparison = df.copy()
                    comparison["Original"] = orig_probs
                    comparison["Pruned"] = pruned_probs

                    fig = px.scatter(
                        comparison,
                        x="Original",
                        y="Pruned",
                        color="Zip Code",
                        title="Original vs Pruned Predictions"
                    )
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines",
                        name="No Change",
                        line=dict(dash="dash")
                    ))
                    st.plotly_chart(fig, width='stretch')

                    # Accuracy comparison
                    orig_preds = (np.array(orig_probs) > 0.5).astype(int)
                    pruned_preds = (np.array(pruned_probs) > 0.5).astype(int)
                    true_labels = df['Target'].values

                    orig_acc = (orig_preds == true_labels).mean()
                    pruned_acc = (pruned_preds == true_labels).mean()

                    st.write(f"**Original Accuracy:** {orig_acc:.4f}")
                    st.write(f"**Pruned Accuracy:** {pruned_acc:.4f}")
                    st.write(f"**Accuracy Drop:** {orig_acc - pruned_acc:.4f}")


# ============================================================================
# MNIST Demo
# ============================================================================
elif demo_mode == "MNIST (Trained Models)":
    st.header("MNIST Model Analysis")
    st.markdown("""
    Analyze trained MLP models on the MNIST dataset using the generic
    interpretability tools from `src.interpretability`.
    """)

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
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
        return model

    try:
        is_sparse = "Sparse" in selected_model
        model = load_mnist_model(str(model_path), is_sparse)
        st.success(f"Loaded {selected_model}")

        # Show model architecture
        st.subheader("Model Architecture")
        manager = HookManager(model)
        layers = manager.list_layers()

        for layer in layers:
            module = manager.get_layer(layer)
            st.write(f"- `{layer}`: {type(module).__name__}")

        # Load sample MNIST data
        st.subheader("Sample Analysis")

        from src.dataset import load_mnist

        @st.cache_data
        def get_mnist_samples(n: int = 10):
            train_loader, test_loader = load_mnist()
            images, labels = next(iter(test_loader))
            return images[:n], labels[:n]

        images, labels = get_mnist_samples(10)

        sample_idx = st.slider("Select sample", 0, 9, 0)
        sample_image = images[sample_idx:sample_idx + 1].view(1, -1)
        sample_label = labels[sample_idx].item()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**True Label**: {sample_label}")
            fig = px.imshow(
                images[sample_idx].squeeze().numpy(),
                color_continuous_scale='gray',
                title=f"Sample {sample_idx}"
            )
            st.plotly_chart(fig, width='stretch')

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
                title="Class Probabilities"
            )
            st.plotly_chart(fig, width='stretch')

        # Layer activations
        st.subheader("Layer Activations")

        # Filter to main layers (Linear layers in Sequential)
        analysis_layers = [l for l in layers if 'network' in l]

        if analysis_layers:
            selected_layer = st.selectbox("Select layer", analysis_layers)

            activations = get_all_activations(model, sample_image, [selected_layer])

            if selected_layer in activations:
                acts = activations[selected_layer].flatten()
                fig = go.Figure(data=go.Bar(
                    x=list(range(len(acts))),
                    y=acts.numpy(),
                    name=selected_layer
                ))
                fig.update_layout(
                    xaxis_title="Neuron Index",
                    yaxis_title="Activation",
                    title=f"Activations in {selected_layer}"
                )
                st.plotly_chart(fig, width='stretch')

                # Top neurons
                top_neurons = find_top_k_neurons(
                    model, sample_image, selected_layer, k=10, method="activation"
                )
                st.write("**Top 10 Active Neurons:**")
                for n in top_neurons:
                    st.write(f"  Neuron {n.neuron_index}: {n.activation_mean:.4f}")

    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.info("Please train the model first using the training scripts.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
