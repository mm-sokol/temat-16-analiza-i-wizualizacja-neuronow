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
    safety_pruning_on_images_tab,
)
from src.streamlit_tabs.ablation import ablation_tab, ablation_on_images_tab

st.set_page_config(page_title="Neural Microscope", layout="wide")
st.title("Neural Microscope: Mechanistic Interpretability Tool")
st.sidebar.header("Configuration")

demo_mode = st.sidebar.selectbox(
    "Demo Mode",
    ["Credit Scoring (Synthetic)", "Credit Score (Real Dataset)", "MNIST (Trained Models)"],
    index=0,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {device.upper()}")


# ============================================================================
# Credit Scoring Demo
# ============================================================================
if demo_mode == "Credit Scoring (Synthetic)":
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
    input_features = sample[["Income", "Credit History", "Age", "Zip Code", "Random Noise"]]
    target = sample["Target"]

    input_tensor = torch.tensor(input_features.values, dtype=torch.float32).unsqueeze(0).to(device)

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
        st.write(f"Zip Code: {zip_val} ({'Advantaged' if zip_val == 1 else 'Disadvantaged'})")

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
        layer_names = [name for name in manager.list_layers() if "layer" in name or "fc" in name]
        activations = get_all_activations(model, sample_image, layer_names)

        # ========== Tab 1: Activations ==========
        with tab1:
            activations_tab(model, sample_image, activations, layer_names, layer_prefix="fc")

        # ========== Tab 2: Bias Circuit Detection ==========
        with tab2:
            # st.error(f"Not implemented yet.")
            circut_detection_on_images_tab(
                model,
                images[sample_idx],
                sample_label,
                layer_names=layer_names,
                layer_prefix="fc",
            )

        # ========== Tab 3: Ablation Study ==========
        with tab3:
            test_loader = load_mnist_data("test")
            ablation_on_images_tab(model, test_loader, activations, layer_names)
            # ablation_tab(model, df, activations, layer_names)

        # ========== Tab 4: Safety Pruning ==========
        with tab4:
            # st.error(f"Not implemented yet.")
            safety_pruning_on_images_tab(model, images[sample_idx], test_loader)

    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.info("Please train the model first using the training scripts.")
    except Exception as e:
        st.error(f"Error loading model: {e}")


# ============================================================================
# Real Credit Score Dataset Demo
# ============================================================================
elif demo_mode == "Credit Score (Real Dataset)":
    st.markdown(
        """
    **Real Credit Score Dataset Analysis**

    This demo uses the Kaggle Credit Score Classification dataset to detect and analyze
    bias in a neural network trained on real financial data. You can select any
    **protected attribute** (e.g., Occupation, Age Group) and analyze which neurons
    encode that information.
    """
    )

    # Import data loader
    from credit_score_data import (
        load_credit_score_dataset,
        get_protected_attribute_values,
        create_contrastive_samples,
        PROTECTED_ATTRIBUTES,
    )

    # Sidebar controls
    max_samples = st.sidebar.slider("Max Samples", 1000, 10000, 5000, step=1000)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)

    if st.sidebar.button("Reload Dataset"):
        st.cache_data.clear()
        st.cache_resource.clear()

    # Load data
    @st.cache_data
    def load_data_cached(max_samples: int, seed: int):
        return load_credit_score_dataset(max_samples=max_samples, random_state=seed)

    try:
        with st.spinner("Loading and preprocessing dataset..."):
            df, X, y, encodings, info = load_data_cached(max_samples, seed)

        st.success(f"Loaded {info.num_samples} samples with {info.num_features} features")

        # Show dataset info
        with st.expander("Dataset Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Numerical Features:**")
                for f in info.numerical_features:
                    st.write(f"  - {f}")
            with col2:
                st.write("**Categorical Features:**")
                for f in info.categorical_features:
                    st.write(f"  - {f}")

            st.write("**Target Distribution:**")
            target_dist = df["Credit_Score"].value_counts().sort_index()
            target_names = {0: "Poor", 1: "Standard", 2: "Good"}
            for idx, count in target_dist.items():
                st.write(f"  - {target_names[idx]}: {count} ({count/len(df)*100:.1f}%)")

        # Protected attribute selection
        st.sidebar.header("Bias Analysis")
        protected_attr = st.sidebar.selectbox(
            "Protected Attribute",
            PROTECTED_ATTRIBUTES,
            index=0,
        )

        # Get available values for this attribute
        attr_values = get_protected_attribute_values(df, protected_attr)

        if len(attr_values) < 2:
            st.warning(f"Not enough values for {protected_attr} to compare.")
        else:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                value_a = st.selectbox("Group A", attr_values, index=0)
            with col2:
                other_values = [v for v in attr_values if v != value_a]
                value_b = st.selectbox("Group B", other_values, index=0)

            st.sidebar.write(f"Comparing: **{value_a}** vs **{value_b}**")

        # Model definition for this dataset
        class CreditScoreModel(torch.nn.Module):
            def __init__(self, input_size: int, hidden_sizes: list, output_size: int = 3):
                super().__init__()
                self.layer1 = torch.nn.Linear(input_size, hidden_sizes[0])
                self.relu1 = torch.nn.ReLU()
                self.layer2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
                self.relu2 = torch.nn.ReLU()
                self.output_layer = torch.nn.Linear(hidden_sizes[1], output_size)

            def forward(self, x):
                x = self.relu1(self.layer1(x))
                x = self.relu2(self.layer2(x))
                x = self.output_layer(x)
                return x

        # Train model
        @st.cache_resource
        def train_credit_model(_X, _y, input_size, epochs=100):
            model = CreditScoreModel(input_size, [64, 32], output_size=3)
            model.to(device)

            X_train = _X.to(device)
            y_train = _y.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

            model.eval()
            return model

        with st.spinner("Training model..."):
            model = train_credit_model(X, y, X.shape[1], epochs=100)

        # Calculate accuracy
        with torch.no_grad():
            outputs = model(X.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            accuracy = (predictions == y).float().mean().item()

        st.write(f"**Model Accuracy:** {accuracy:.2%}")

        # Create tabs for analysis
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Dataset & Bias Overview",
                "Neuron Activations",
                "Circuit Discovery",
                "Ablation & Pruning",
            ]
        )

        # ========== Tab 1: Dataset & Bias Overview ==========
        with tab1:
            st.header("Bias Overview by Protected Attribute")

            # Show outcome distribution by protected attribute
            if protected_attr in df.columns:
                st.subheader(f"Credit Score Distribution by {protected_attr}")

                # Create cross-tabulation
                cross_tab = (
                    pd.crosstab(df[protected_attr], df["Credit_Score"], normalize="index") * 100
                )

                cross_tab.columns = ["Poor", "Standard", "Good"]

                fig = px.bar(
                    cross_tab.reset_index(),
                    x=protected_attr,
                    y=["Poor", "Standard", "Good"],
                    title=f"Credit Score Distribution by {protected_attr} (%)",
                    barmode="group",
                )
                st.plotly_chart(fig, width="stretch")

                # Model predictions by group
                st.subheader("Model Predictions by Group")

                df_with_preds = df.copy()
                with torch.no_grad():
                    preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
                df_with_preds["Prediction"] = preds

                pred_by_group = (
                    df_with_preds.groupby(protected_attr)["Prediction"]
                    .apply(
                        lambda x: pd.Series(
                            {
                                "Poor": (x == 0).mean() * 100,
                                "Standard": (x == 1).mean() * 100,
                                "Good": (x == 2).mean() * 100,
                            }
                        )
                    )
                    .unstack()
                )

                fig2 = px.bar(
                    pred_by_group.reset_index(),
                    x=protected_attr,
                    y=["Poor", "Standard", "Good"],
                    title=f"Model Predictions by {protected_attr} (%)",
                    barmode="group",
                )
                st.plotly_chart(fig2, width="stretch")

                # Bias metric
                if len(attr_values) >= 2:
                    good_rate_a = (
                        pred_by_group.loc[value_a, "Good"] if value_a in pred_by_group.index else 0
                    )
                    good_rate_b = (
                        pred_by_group.loc[value_b, "Good"] if value_b in pred_by_group.index else 0
                    )

                    st.metric(
                        "Bias Gap (Good prediction rate)",
                        f"{abs(good_rate_a - good_rate_b):.1f}%",
                        delta=f"{value_a}: {good_rate_a:.1f}% vs {value_b}: {good_rate_b:.1f}%",
                    )

        # ========== Tab 2: Neuron Activations ==========
        with tab2:
            st.header("Neuron Activation Analysis")

            manager = HookManager(model)
            layer_names = [name for name in manager.list_layers() if "layer" in name]

            # Sample selection
            sample_idx = st.slider("Select Sample", 0, len(df) - 1, 0)
            sample_x = X[sample_idx : sample_idx + 1].to(device)

            st.write(
                f"**Sample {sample_idx}:** {protected_attr} = {df.iloc[sample_idx][protected_attr]}"
            )

            # Get activations
            activations = get_all_activations(model, sample_x, layer_names)

            col1, col2 = st.columns(2)

            with col1:
                if "layer1" in activations:
                    acts = activations["layer1"].cpu().numpy().flatten()
                    st.subheader(f"Layer 1 ({len(acts)} neurons)")
                    fig = px.bar(
                        x=list(range(len(acts))), y=acts, labels={"x": "Neuron", "y": "Activation"}
                    )
                    st.plotly_chart(fig, width="stretch")

            with col2:
                if "layer2" in activations:
                    acts = activations["layer2"].cpu().numpy().flatten()
                    st.subheader(f"Layer 2 ({len(acts)} neurons)")
                    fig = px.bar(
                        x=list(range(len(acts))), y=acts, labels={"x": "Neuron", "y": "Activation"}
                    )
                    st.plotly_chart(fig, width="stretch")

            # Compare two groups
            st.subheader(f"Activation Difference: {value_a} vs {value_b}")

            mask_a = df[protected_attr] == value_a
            mask_b = df[protected_attr] == value_b

            if mask_a.sum() > 0 and mask_b.sum() > 0:
                # Get mean activations for each group
                X_a = X[mask_a.values].to(device)
                X_b = X[mask_b.values].to(device)

                acts_a = get_all_activations(model, X_a, layer_names)
                acts_b = get_all_activations(model, X_b, layer_names)

                for layer_name in layer_names:
                    if layer_name in acts_a and layer_name in acts_b:
                        mean_a = acts_a[layer_name].mean(dim=0).cpu().numpy()
                        mean_b = acts_b[layer_name].mean(dim=0).cpu().numpy()
                        diff = mean_a - mean_b

                        st.write(f"**{layer_name}:** Mean activation difference")
                        fig = px.bar(
                            x=list(range(len(diff))),
                            y=diff,
                            labels={"x": "Neuron", "y": f"Mean({value_a}) - Mean({value_b})"},
                            title=f"Neurons sensitive to {protected_attr}",
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig, width="stretch")

        # ========== Tab 3: Circuit Discovery ==========
        with tab3:
            st.header("Bias Circuit Discovery")
            st.markdown(
                f"""
            Finding neurons that encode **{protected_attr}** by comparing activations
            between **{value_a}** and **{value_b}** samples.
            """
            )

            if st.button("Discover Bias Circuit"):
                mask_a = df[protected_attr] == value_a
                mask_b = df[protected_attr] == value_b

                if mask_a.sum() > 0 and mask_b.sum() > 0:
                    # Take representative samples
                    idx_a = mask_a.values.nonzero()[0][0]
                    idx_b = mask_b.values.nonzero()[0][0]

                    x_a = X[idx_a : idx_a + 1].to(device)
                    x_b = X[idx_b : idx_b + 1].to(device)

                    circuit = discover_circuits(model, x_a, x_b, layer_names, top_k=5)

                    st.write(f"**Total Circuit Importance:** {circuit.total_importance:.4f}")
                    st.write("**Identified Bias Neurons:**")

                    for neuron in circuit.neurons[:15]:
                        st.write(
                            f"- **{neuron.layer_name}** Neuron {neuron.neuron_index}: "
                            f"importance = {neuron.importance_score:.4f}"
                        )

                    st.session_state["bias_circuit"] = circuit
                    st.session_state["bias_attr"] = protected_attr
                    st.success("Circuit saved! Go to 'Ablation & Pruning' tab.")
                else:
                    st.error("Not enough samples in one of the groups.")

        # ========== Tab 4: Ablation & Pruning ==========
        with tab4:
            st.header("Ablation Study & Safety Pruning")

            if "bias_circuit" not in st.session_state:
                st.warning("Please discover the bias circuit first (Tab 3).")
            else:
                circuit = st.session_state["bias_circuit"]

                # Group neurons by layer
                layer_neurons = {}
                for n in circuit.neurons:
                    if n.layer_name not in layer_neurons:
                        layer_neurons[n.layer_name] = []
                    layer_neurons[n.layer_name].append(n.neuron_index)

                st.subheader("Discovered Bias Neurons")
                for layer, neurons in layer_neurons.items():
                    st.write(f"**{layer}:** {neurons}")

                # Ablation study
                st.subheader("Ablation Study")
                ablate_layer = st.selectbox("Layer to ablate", list(layer_neurons.keys()))

                if ablate_layer:
                    neurons_to_ablate = st.multiselect(
                        "Neurons to ablate",
                        layer_neurons[ablate_layer],
                        default=layer_neurons[ablate_layer][:3],
                    )

                    if neurons_to_ablate and st.button("Run Ablation"):
                        mask_a = df[protected_attr] == value_a
                        mask_b = df[protected_attr] == value_b

                        # Get predictions before and after ablation
                        results = []

                        for idx in range(min(200, len(df))):
                            x = X[idx : idx + 1].to(device)
                            group = df.iloc[idx][protected_attr]

                            with torch.no_grad():
                                orig_pred = model(x).softmax(dim=1)[0, 2].item()  # P(Good)

                            ablated_out = run_ablation(model, x, ablate_layer, neurons_to_ablate)
                            ablated_pred = ablated_out.softmax(dim=1)[0, 2].item()

                            results.append(
                                {
                                    "Group": group,
                                    "Original": orig_pred,
                                    "Ablated": ablated_pred,
                                    "Diff": ablated_pred - orig_pred,
                                }
                            )

                        results_df = pd.DataFrame(results)

                        # Show scatter plot
                        fig = px.scatter(
                            results_df,
                            x="Original",
                            y="Ablated",
                            color="Group",
                            title="Original vs Ablated P(Good)",
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode="lines",
                                name="No Change",
                                line=dict(dash="dash"),
                            )
                        )
                        st.plotly_chart(fig, width="stretch")

                        # Show effect by group
                        st.subheader("Average Effect by Group")
                        effect_by_group = results_df.groupby("Group")["Diff"].mean()
                        st.bar_chart(effect_by_group)

                # Safety Pruning
                st.subheader("Safety Pruning")
                prune_layer = st.selectbox(
                    "Layer to prune", list(layer_neurons.keys()), key="prune_layer"
                )

                if prune_layer:
                    prune_neurons = st.multiselect(
                        "Neurons to prune permanently",
                        layer_neurons[prune_layer],
                        default=layer_neurons[prune_layer][:2],
                        key="prune_neurons",
                    )

                    if prune_neurons and st.button("Apply Pruning"):
                        pruned_model, result = prune_biased_neurons(
                            model, prune_layer, prune_neurons, copy=True
                        )

                        st.write(result.summary())

                        # Evaluate impact
                        with torch.no_grad():
                            orig_preds = model(X.to(device)).argmax(dim=1).cpu()
                            pruned_preds = pruned_model(X.to(device)).argmax(dim=1).cpu()

                        orig_acc = (orig_preds == y).float().mean().item()
                        pruned_acc = (pruned_preds == y).float().mean().item()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Accuracy", f"{orig_acc:.2%}")
                        with col2:
                            st.metric("Pruned Accuracy", f"{pruned_acc:.2%}")
                        with col3:
                            st.metric("Accuracy Drop", f"{orig_acc - pruned_acc:.2%}")

                        # Bias change
                        df_with_preds = df.copy()
                        df_with_preds["Orig_Pred"] = orig_preds.numpy()
                        df_with_preds["Pruned_Pred"] = pruned_preds.numpy()

                        orig_good_a = (
                            df_with_preds[df_with_preds[protected_attr] == value_a]["Orig_Pred"]
                            == 2
                        ).mean()
                        orig_good_b = (
                            df_with_preds[df_with_preds[protected_attr] == value_b]["Orig_Pred"]
                            == 2
                        ).mean()
                        pruned_good_a = (
                            df_with_preds[df_with_preds[protected_attr] == value_a]["Pruned_Pred"]
                            == 2
                        ).mean()
                        pruned_good_b = (
                            df_with_preds[df_with_preds[protected_attr] == value_b]["Pruned_Pred"]
                            == 2
                        ).mean()

                        orig_gap = abs(orig_good_a - orig_good_b)
                        pruned_gap = abs(pruned_good_a - pruned_good_b)

                        st.subheader("Fairness Impact")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Bias Gap", f"{orig_gap:.2%}")
                        with col2:
                            st.metric(
                                "Pruned Bias Gap",
                                f"{pruned_gap:.2%}",
                                delta=f"{pruned_gap - orig_gap:.2%}",
                                delta_color="inverse",
                            )

    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.info("Please ensure the Credit Score dataset is in data/raw/CreditScore/")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback

        st.code(traceback.format_exc())
