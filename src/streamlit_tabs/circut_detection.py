import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))
from src.interpretability import (
    HookManager,
    find_top_k_neurons,
)
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


def circut_detection_tab(model, input_tensor, input_features, layer_names):
    st.header("Detecting the Discrimination Circuit")
    st.markdown(
        """
        **Experiment:** What if we change *only* the Zip Code for this applicant?
        We flip the Zip Code (0 -> 1 or 1 -> 0) and observe which neurons change
        their activation the most. These neurons form the **Bias Circuit**.
        """
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Counterfactual
    cf_features = input_features.copy()
    cf_features["Zip Code"] = 1 - cf_features["Zip Code"]  # Flip
    cf_tensor = (
        torch.tensor(cf_features.values, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # Get activations for both using generic module
    orig_activations = get_all_activations(model, input_tensor, layer_names)
    cf_activations = get_all_activations(model, cf_tensor, layer_names)

    # Calculate Differences
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if "layer1" in orig_activations:
            l1_acts = orig_activations["layer1"].cpu().numpy().flatten()
            cf_l1_acts = cf_activations["layer1"].cpu().numpy().flatten()
            diff_l1 = cf_l1_acts - l1_acts
            st.subheader("Layer 1 Sensitivity (Bias Neurons)")
            fig_d1 = px.bar(
                x=list(range(len(diff_l1))),
                y=diff_l1,
                labels={"x": "Neuron Index", "y": "Change in Activation"},
                title="Change when Zip Code is Flipped",
            )
            st.plotly_chart(fig_d1, width="stretch")

    with col_d2:
        if "layer2" in orig_activations:
            l2_acts = orig_activations["layer2"].cpu().numpy().flatten()
            cf_l2_acts = cf_activations["layer2"].cpu().numpy().flatten()
            diff_l2 = cf_l2_acts - l2_acts
            st.subheader("Layer 2 Sensitivity (Bias Neurons)")
            fig_d2 = px.bar(
                x=list(range(len(diff_l2))),
                y=diff_l2,
                labels={"x": "Neuron Index", "y": "Change in Activation"},
                title="Change when Zip Code is Flipped",
            )
            st.plotly_chart(fig_d2, width="stretch")

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
        st.session_state["bias_circuit"] = circuit
        st.success("Circuit saved! Go to 'Safety Pruning' tab to apply interventions.")


def get_couterfactual(
    model,
    original_img,
    target_label,
    optim_steps,
    l2_reg_param=0.01,
    l1_reg_param=0.005,
):

    couterfactual_img = original_img.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([couterfactual_img], lr=1e-2)

    for _ in range(optim_steps):
        optimizer.zero_grad()

        logits = model(couterfactual_img.view(1, -1))

        orig_class = logits.argmax(dim=1).item()

        m = 0.2
        target_loss = torch.relu(
            logits[:, orig_class] - logits[:, target_label] + m
        ).mean()

        # target_loss = -logits[:, target_label].mean()

        proximity_loss = torch.norm(couterfactual_img - original_img, p=2)

        # 1 pixel shift in the horizontal direction
        # 1 pixel shift in the hertical direction
        # regularization_loss = torch.mean(
        #     torch.abs(couterfactual_img[:, :, :-1] - couterfactual_img[:, :, 1:])
        # ) + torch.mean(
        #     torch.abs(couterfactual_img[:, :-1, :] - couterfactual_img[:, 1:, :])
        # )
        regularization_loss = torch.norm(couterfactual_img - original_img, p=1)

        loss = (
            target_loss
            + l2_reg_param * proximity_loss
            + l1_reg_param * regularization_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            couterfactual_img.clamp_(0, 1)

    return couterfactual_img


def circut_detection_on_images_tab(
    model,
    original_img,
    original_label,
    layer_names,
    optim_steps=300,
    layer_prefix="layer",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    col1, col2 = st.columns(2)

    with col1:

        target_label = st.radio(
            "Select target class for counterfactual: ",
            options=list(range(1, 11)),
            horizontal=True,
            key="target_label",
        )
        target_label -= 1

        st.session_state.couterfactual = get_couterfactual(
            model,
            original_img,
            target_label,
            optim_steps,
        )

        couterfactual_img = st.session_state.couterfactual
        couterfactual_img = couterfactual_img.detach().to(device)

        orig_activations = get_all_activations(
            model, original_img.view(1, -1), layer_names
        )
        cf_activations = get_all_activations(
            model, couterfactual_img.view(1, -1), layer_names
        )

    with col2:
        st.image(
            couterfactual_img.view(28, 28).squeeze().detach().cpu().numpy(),
            caption="Counterfactual image",
            clamp=True,
        )

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if f"{layer_prefix}1" in orig_activations:
            l1_acts = orig_activations[f"{layer_prefix}1"].cpu().numpy().flatten()
            cf_l1_acts = cf_activations[f"{layer_prefix}1"].cpu().numpy().flatten()
            diff_l1 = cf_l1_acts - l1_acts
            st.subheader("Layer 1 Sensitivity (Bias Neurons)")
            fig_d1 = px.bar(
                x=list(range(len(diff_l1))),
                y=diff_l1,
                labels={"x": "Neuron Index", "y": "Change in Activation"},
                title="Activation change after switching to couterfactual",
            )
            st.plotly_chart(fig_d1, width="stretch")

    with col_d2:
        if f"{layer_prefix}2" in orig_activations:
            l2_acts = orig_activations[f"{layer_prefix}2"].cpu().numpy().flatten()
            cf_l2_acts = cf_activations[f"{layer_prefix}2"].cpu().numpy().flatten()
            diff_l2 = cf_l2_acts - l2_acts
            st.subheader("Layer 2 Sensitivity (Bias Neurons)")
            fig_d2 = px.bar(
                x=list(range(len(diff_l2))),
                y=diff_l2,
                labels={"x": "Neuron Index", "y": "Change in Activation"},
                title="Activation change after switching to couterfactual",
            )
            st.plotly_chart(fig_d2, width="stretch")

    st.info(
        "Neurons with large bars are strongly changed by passing couterfactual image."
        "These are candidates for pruning."
    )

    # Circuit discovery using generic module
    st.subheader("Automated Circuit Discovery")
    if st.button("Discover Bias Circuit"):
        circuit = discover_circuits(
            model, original_img, couterfactual_img, layer_names, top_k=5
        )
        st.write(f"**Total Circuit Importance:** {circuit.total_importance:.4f}")
        st.write("**Identified Neurons:**")
        for neuron in circuit.neurons[:10]:
            st.write(
                f"- **{neuron.layer_name}** Neuron {neuron.neuron_index}: "
                f"importance={neuron.importance_score:.4f}"
            )
        st.session_state["bias_circuit"] = circuit
        st.success("Circuit saved! Go to 'Safety Pruning' tab to apply interventions.")


def safety_pruning(model, input_features, df):
    st.header("Safety Pruning")
    st.info("Permanently remove biased neurons using src.interpretability.pruning")
    st.markdown(
        """
        **Goal:** Remove neurons that encode protected attributes (Zip Code)
        to improve model fairness while minimizing accuracy loss.
        """
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "bias_circuit" not in st.session_state:
        st.warning(
            "No bias circuit discovered yet. Go to 'Bias Circuit Detection' tab first."
        )
    else:
        circuit = st.session_state["bias_circuit"]
        layer_neurons = {}
        for n in circuit.neurons:
            if n.layer_name not in layer_neurons:
                layer_neurons[n.layer_name] = []
            layer_neurons[n.layer_name].append((n.neuron_index, n.importance_score))

        st.subheader("Discovered Bias Neurons")
        for layer, neurons in layer_neurons.items():
            st.write(f"**{layer}:** {[n[0] for n in neurons]}")

        prune_layer = st.selectbox(
            "Layer to prune", list(layer_neurons.keys()), key="prune_layer"
        )

        if prune_layer:
            available_neurons = [n[0] for n in layer_neurons[prune_layer]]
            prune_neurons = st.multiselect(
                "Neurons to prune",
                available_neurons,
                default=(
                    available_neurons[:2]
                    if len(available_neurons) >= 2
                    else available_neurons
                ),
                key="prune_neurons",
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
                test_features_adv["Zip Code"] = 1
                test_features_dis = input_features.copy()
                test_features_dis["Zip Code"] = 0

                x_adv = (
                    torch.tensor(test_features_adv.values, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                x_dis = (
                    torch.tensor(test_features_dis.values, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
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
                        delta=f"{pruned_adv - orig_adv:.3f}",
                    )
                with col2:
                    st.metric(
                        "Disadvantaged Zip",
                        f"{pruned_dis:.3f}",
                        delta=f"{pruned_dis - orig_dis:.3f}",
                    )
                with col3:
                    old_gap = orig_adv - orig_dis
                    new_gap = pruned_adv - pruned_dis
                    st.metric(
                        "Bias Gap",
                        f"{new_gap:.3f}",
                        delta=f"{new_gap - old_gap:.3f}",
                        delta_color="inverse",
                    )
                # Full dataset comparison
                st.subheader("Full Dataset Impact")
                orig_probs = []
                pruned_probs = []
                for idx, row in df.iterrows():
                    x = (
                        torch.tensor(
                            row[
                                [
                                    "Income",
                                    "Credit History",
                                    "Age",
                                    "Zip Code",
                                    "Random Noise",
                                ]
                            ].values,
                            dtype=torch.float32,
                        )
                        .unsqueeze(0)
                        .to(device)
                    )
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
                    title="Original vs Pruned Predictions",
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
                # Accuracy comparison
                orig_preds = (np.array(orig_probs) > 0.5).astype(int)
                pruned_preds = (np.array(pruned_probs) > 0.5).astype(int)
                true_labels = df["Target"].values
                orig_acc = (orig_preds == true_labels).mean()
                pruned_acc = (pruned_preds == true_labels).mean()
                st.write(f"**Original Accuracy:** {orig_acc:.4f}")
                st.write(f"**Pruned Accuracy:** {pruned_acc:.4f}")
                st.write(f"**Accuracy Drop:** {orig_acc - pruned_acc:.4f}")
