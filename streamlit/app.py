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
    [
        "Credit Scoring (Synthetic)",
        "Credit Score (Real Dataset)",
        "MNIST (Trained Models)"
    ],
    index=0,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.write(f"Device: {device.upper()}")


# ============================================================================
# Credit Scoring Demo
# ============================================================================
if demo_mode == "Credit Scoring (Synthetic)":
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Activations",
        "Bias Circuit Detection",
        "Ablation Study",
        "Safety Pruning",
        "Experiments"
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

    # ========== Tab 5: Experiments ==========
    with tab5:
        st.header("Validation Experiments")
        st.markdown("""
        **Purpose**: Before trusting interpretability findings, we must validate that our tools
        actually work. These experiments use the synthetic data where we **know the ground truth**
        (Zip Code weight = 1.5, Income = 0.5, Noise = 0.0).
        """)

        experiment = st.selectbox(
            "Select Experiment",
            ["Study 1: Multi-Seed Consistency", "Study 2: Feature Attribution", "Study 3: Circuit Completeness"]
        )

        # ------ Study 1: Multi-Seed Consistency ------
        if experiment == "Study 1: Multi-Seed Consistency":
            st.subheader("Do the same bias neurons appear across different training runs?")
            st.markdown("""
            If our tools are reliable, they should identify **consistent** neurons across models
            trained with different random seeds. Random results = unreliable tools.
            """)

            n_runs = st.slider("Number of training runs", 3, 10, 5)
            threshold = st.slider("Activation difference threshold", 0.1, 1.0, 0.3)

            if st.button("Run Multi-Seed Experiment", key="study1"):
                results = {"layer1": {}, "layer2": {}}
                progress = st.progress(0)

                for run_idx in range(n_runs):
                    # Train model with different seed
                    run_seed = seed + run_idx * 100
                    run_df = generate_synthetic_data(n_samples=n_samples, seed=run_seed)
                    run_model = train_model(run_df, epochs=200, device=device)

                    # Get a sample and its counterfactual
                    features = ['Income', 'CreditHistory', 'Age', 'ZipCode', 'RandomNoise']
                    sample = run_df[features].iloc[0].values.astype(np.float32)
                    cf_sample = sample.copy()
                    cf_sample[3] = 1 - cf_sample[3]  # Flip ZipCode

                    sample_t = torch.tensor(sample).unsqueeze(0).to(device)
                    cf_t = torch.tensor(cf_sample).unsqueeze(0).to(device)

                    # Get activations for both
                    layer_names = ['layer1', 'layer2']
                    orig_acts = get_all_activations(run_model, sample_t, layer_names)
                    cf_acts = get_all_activations(run_model, cf_t, layer_names)

                    # Find neurons with large activation difference
                    for layer in layer_names:
                        diff = (cf_acts[layer] - orig_acts[layer]).abs().cpu().numpy().flatten()
                        biased_neurons = np.where(diff > threshold)[0]
                        for neuron in biased_neurons:
                            if neuron not in results[layer]:
                                results[layer][neuron] = 0
                            results[layer][neuron] += 1

                    progress.progress((run_idx + 1) / n_runs)

                # Display results
                st.success(f"Completed {n_runs} training runs")

                for layer in ['layer1', 'layer2']:
                    st.subheader(f"{layer} - Neuron Consistency")
                    if results[layer]:
                        neurons = list(results[layer].keys())
                        counts = list(results[layer].values())
                        consistency = [c / n_runs for c in counts]

                        fig = px.bar(
                            x=[f"N{n}" for n in neurons],
                            y=consistency,
                            labels={'x': 'Neuron', 'y': f'Appearance Rate (out of {n_runs} runs)'},
                            title=f"{layer}: How often each neuron was flagged as biased"
                        )
                        fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                                     annotation_text="80% = Reliable")
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary
                        reliable = [n for n, c in zip(neurons, consistency) if c >= 0.8]
                        if reliable:
                            st.success(f"Reliable bias neurons (>=80%): {reliable}")
                        else:
                            st.warning("No neurons appeared in >=80% of runs. Tools may need calibration.")
                    else:
                        st.info(f"No neurons exceeded threshold in {layer}")

        # ------ Study 2: Feature Attribution ------
        elif experiment == "Study 2: Feature Attribution":
            st.subheader("Does ablation correctly rank feature importance?")
            st.markdown("""
            **Ground truth weights**: ZipCode=1.5, Income=0.5, CreditHistory=0.3, Age=0.2, Noise=0.0

            We test: for each feature, which neurons encode it? Ablating those neurons should
            affect predictions proportionally to the feature's true importance.
            """)

            if st.button("Run Feature Attribution Study", key="study2"):
                features = ['Income', 'CreditHistory', 'Age', 'ZipCode', 'RandomNoise']
                true_weights = {'Income': 0.5, 'CreditHistory': 0.3, 'Age': 0.2, 'ZipCode': 1.5, 'RandomNoise': 0.0}

                feature_importance = {}
                progress = st.progress(0)

                for feat_idx, feat_name in enumerate(features):
                    # Create paired samples: one with feature=0, one with feature=1 (normalized)
                    base_sample = df[features].iloc[0].values.astype(np.float32)
                    modified_sample = base_sample.copy()

                    # Flip the feature value
                    feat_min = df[feat_name].min()
                    feat_max = df[feat_name].max()
                    if base_sample[feat_idx] < (feat_min + feat_max) / 2:
                        modified_sample[feat_idx] = feat_max
                    else:
                        modified_sample[feat_idx] = feat_min

                    base_t = torch.tensor(base_sample).unsqueeze(0).to(device)
                    mod_t = torch.tensor(modified_sample).unsqueeze(0).to(device)

                    # Get activations
                    layer_names = ['layer1', 'layer2']
                    base_acts = get_all_activations(model, base_t, layer_names)
                    mod_acts = get_all_activations(model, mod_t, layer_names)

                    # Find neurons most affected by this feature
                    total_diff = 0
                    for layer in layer_names:
                        diff = (mod_acts[layer] - base_acts[layer]).abs().sum().item()
                        total_diff += diff

                    feature_importance[feat_name] = total_diff
                    progress.progress((feat_idx + 1) / len(features))

                # Normalize and compare to ground truth
                max_imp = max(feature_importance.values()) if max(feature_importance.values()) > 0 else 1
                normalized_imp = {k: v / max_imp for k, v in feature_importance.items()}

                max_true = max(true_weights.values())
                normalized_true = {k: v / max_true for k, v in true_weights.items()}

                # Plot comparison
                comparison_df = pd.DataFrame({
                    'Feature': features,
                    'Measured Importance': [normalized_imp[f] for f in features],
                    'True Weight': [normalized_true[f] for f in features]
                })

                fig = px.bar(
                    comparison_df.melt(id_vars='Feature', var_name='Type', value_name='Value'),
                    x='Feature', y='Value', color='Type', barmode='group',
                    title='Feature Importance: Measured vs Ground Truth'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Compute rank correlation
                measured_rank = sorted(features, key=lambda f: normalized_imp[f], reverse=True)
                true_rank = sorted(features, key=lambda f: normalized_true[f], reverse=True)

                st.subheader("Ranking Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Measured Ranking:**")
                    for i, f in enumerate(measured_rank):
                        st.write(f"{i+1}. {f}")
                with col2:
                    st.write("**True Ranking:**")
                    for i, f in enumerate(true_rank):
                        st.write(f"{i+1}. {f}")

                # Check if ZipCode is #1 and Noise is last
                if measured_rank[0] == 'ZipCode':
                    st.success("ZipCode correctly identified as most important!")
                else:
                    st.error(f"Expected ZipCode first, got {measured_rank[0]}")

                if measured_rank[-1] == 'RandomNoise':
                    st.success("RandomNoise correctly identified as least important!")
                else:
                    st.warning(f"Expected RandomNoise last, got {measured_rank[-1]}")

        # ------ Study 3: Circuit Completeness ------
        elif experiment == "Study 3: Circuit Completeness":
            st.subheader("Is the discovered bias circuit the COMPLETE explanation?")
            st.markdown("""
            We find neurons that respond to ZipCode, then ablate ALL of them.
            If discrimination persists, there's another pathway we missed.
            """)

            threshold = st.slider("Bias detection threshold", 0.1, 0.5, 0.2, key="study3_thresh")

            if st.button("Run Completeness Test", key="study3"):
                features = ['Income', 'CreditHistory', 'Age', 'ZipCode', 'RandomNoise']

                # First, find all bias neurons
                st.write("**Step 1: Finding bias neurons across dataset...**")
                bias_neurons = {'layer1': set(), 'layer2': set()}

                sample_size = min(100, len(df))
                progress = st.progress(0)

                for i in range(sample_size):
                    sample = df[features].iloc[i].values.astype(np.float32)
                    cf_sample = sample.copy()
                    cf_sample[3] = 1 - cf_sample[3]  # Flip ZipCode

                    sample_t = torch.tensor(sample).unsqueeze(0).to(device)
                    cf_t = torch.tensor(cf_sample).unsqueeze(0).to(device)

                    layer_names = ['layer1', 'layer2']
                    orig_acts = get_all_activations(model, sample_t, layer_names)
                    cf_acts = get_all_activations(model, cf_t, layer_names)

                    for layer in layer_names:
                        diff = (cf_acts[layer] - orig_acts[layer]).abs().cpu().numpy().flatten()
                        biased = np.where(diff > threshold)[0]
                        bias_neurons[layer].update(biased)

                    progress.progress((i + 1) / sample_size)

                st.write(f"Found bias neurons - Layer1: {sorted(bias_neurons['layer1'])}, Layer2: {sorted(bias_neurons['layer2'])}")

                # Step 2: Measure discrimination before ablation
                st.write("**Step 2: Measuring discrimination before ablation...**")
                zip0_preds, zip1_preds = [], []

                for i in range(len(df)):
                    sample = df[features].iloc[i].values.astype(np.float32)
                    sample_t = torch.tensor(sample).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = torch.sigmoid(model(sample_t)).item()
                    if df['ZipCode'].iloc[i] == 0:
                        zip0_preds.append(pred)
                    else:
                        zip1_preds.append(pred)

                orig_gap = abs(np.mean(zip1_preds) - np.mean(zip0_preds))
                st.write(f"Original discrimination gap: **{orig_gap:.4f}**")

                # Step 3: Ablate all bias neurons and re-measure
                st.write("**Step 3: Ablating all bias neurons...**")

                # Convert to list format for pruning
                neurons_to_prune = []
                for layer, neurons in bias_neurons.items():
                    for n in neurons:
                        neurons_to_prune.append((layer, n))

                if neurons_to_prune:
                    pruned_model = prune_model(model, neurons_to_prune)

                    zip0_preds_pruned, zip1_preds_pruned = [], []
                    for i in range(len(df)):
                        sample = df[features].iloc[i].values.astype(np.float32)
                        sample_t = torch.tensor(sample).unsqueeze(0).to(device)
                        with torch.no_grad():
                            pred = torch.sigmoid(pruned_model(sample_t)).item()
                        if df['ZipCode'].iloc[i] == 0:
                            zip0_preds_pruned.append(pred)
                        else:
                            zip1_preds_pruned.append(pred)

                    ablated_gap = abs(np.mean(zip1_preds_pruned) - np.mean(zip0_preds_pruned))
                    reduction = (orig_gap - ablated_gap) / orig_gap * 100 if orig_gap > 0 else 0

                    st.write(f"Discrimination gap after ablation: **{ablated_gap:.4f}**")
                    st.write(f"Reduction: **{reduction:.1f}%**")

                    # Verdict
                    st.subheader("Verdict")
                    if reduction >= 90:
                        st.success(f"Circuit is COMPLETE! Ablating {len(neurons_to_prune)} neurons removed {reduction:.1f}% of discrimination.")
                    elif reduction >= 50:
                        st.warning(f"Circuit is PARTIAL. {reduction:.1f}% reduction - there may be additional pathways.")
                    else:
                        st.error(f"Circuit is INCOMPLETE. Only {reduction:.1f}% reduction - major pathways were missed.")

                    # Accuracy check
                    orig_preds = (np.array(zip0_preds + zip1_preds) > 0.5).astype(int)
                    pruned_preds = (np.array(zip0_preds_pruned + zip1_preds_pruned) > 0.5).astype(int)
                    true_labels = df['Target'].values

                    orig_acc = (orig_preds == true_labels).mean()
                    pruned_acc = (pruned_preds == true_labels).mean()

                    st.write(f"Accuracy: {orig_acc:.4f} → {pruned_acc:.4f} (Δ = {orig_acc - pruned_acc:.4f})")
                else:
                    st.warning("No bias neurons found at this threshold. Try lowering it.")


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


# ============================================================================
# Real Credit Score Dataset Demo
# ============================================================================
elif demo_mode == "Credit Score (Real Dataset)":
    st.markdown("""
    **Real Credit Score Dataset Analysis**

    This demo uses the Kaggle Credit Score Classification dataset to detect and analyze
    bias in a neural network trained on real financial data. You can select any
    **protected attribute** (e.g., Occupation, Age Group) and analyze which neurons
    encode that information.
    """)

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
            target_dist = df['Credit_Score'].value_counts().sort_index()
            target_names = {0: 'Poor', 1: 'Standard', 2: 'Good'}
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dataset & Bias Overview",
            "Neuron Activations",
            "Circuit Discovery",
            "Ablation & Pruning"
        ])

        # ========== Tab 1: Dataset & Bias Overview ==========
        with tab1:
            st.header("Bias Overview by Protected Attribute")

            # Show outcome distribution by protected attribute
            if protected_attr in df.columns:
                st.subheader(f"Credit Score Distribution by {protected_attr}")

                # Create cross-tabulation
                cross_tab = pd.crosstab(
                    df[protected_attr],
                    df['Credit_Score'],
                    normalize='index'
                ) * 100

                cross_tab.columns = ['Poor', 'Standard', 'Good']

                fig = px.bar(
                    cross_tab.reset_index(),
                    x=protected_attr,
                    y=['Poor', 'Standard', 'Good'],
                    title=f"Credit Score Distribution by {protected_attr} (%)",
                    barmode='group'
                )
                st.plotly_chart(fig, width='stretch')

                # Model predictions by group
                st.subheader("Model Predictions by Group")

                df_with_preds = df.copy()
                with torch.no_grad():
                    preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
                df_with_preds['Prediction'] = preds

                pred_by_group = df_with_preds.groupby(protected_attr)['Prediction'].apply(
                    lambda x: pd.Series({
                        'Poor': (x == 0).mean() * 100,
                        'Standard': (x == 1).mean() * 100,
                        'Good': (x == 2).mean() * 100
                    })
                ).unstack()

                fig2 = px.bar(
                    pred_by_group.reset_index(),
                    x=protected_attr,
                    y=['Poor', 'Standard', 'Good'],
                    title=f"Model Predictions by {protected_attr} (%)",
                    barmode='group'
                )
                st.plotly_chart(fig2, width='stretch')

                # Bias metric
                if len(attr_values) >= 2:
                    good_rate_a = pred_by_group.loc[value_a, 'Good'] if value_a in pred_by_group.index else 0
                    good_rate_b = pred_by_group.loc[value_b, 'Good'] if value_b in pred_by_group.index else 0

                    st.metric(
                        "Bias Gap (Good prediction rate)",
                        f"{abs(good_rate_a - good_rate_b):.1f}%",
                        delta=f"{value_a}: {good_rate_a:.1f}% vs {value_b}: {good_rate_b:.1f}%"
                    )

        # ========== Tab 2: Neuron Activations ==========
        with tab2:
            st.header("Neuron Activation Analysis")

            manager = HookManager(model)
            layer_names = [name for name in manager.list_layers() if 'layer' in name]

            # Sample selection
            sample_idx = st.slider("Select Sample", 0, len(df) - 1, 0)
            sample_x = X[sample_idx:sample_idx+1].to(device)

            st.write(f"**Sample {sample_idx}:** {protected_attr} = {df.iloc[sample_idx][protected_attr]}")

            # Get activations
            activations = get_all_activations(model, sample_x, layer_names)

            col1, col2 = st.columns(2)

            with col1:
                if 'layer1' in activations:
                    acts = activations['layer1'].cpu().numpy().flatten()
                    st.subheader(f"Layer 1 ({len(acts)} neurons)")
                    fig = px.bar(x=list(range(len(acts))), y=acts,
                                labels={'x': 'Neuron', 'y': 'Activation'})
                    st.plotly_chart(fig, width='stretch')

            with col2:
                if 'layer2' in activations:
                    acts = activations['layer2'].cpu().numpy().flatten()
                    st.subheader(f"Layer 2 ({len(acts)} neurons)")
                    fig = px.bar(x=list(range(len(acts))), y=acts,
                                labels={'x': 'Neuron', 'y': 'Activation'})
                    st.plotly_chart(fig, width='stretch')

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
                            labels={'x': 'Neuron', 'y': f'Mean({value_a}) - Mean({value_b})'},
                            title=f"Neurons sensitive to {protected_attr}"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig, width='stretch')

        # ========== Tab 3: Circuit Discovery ==========
        with tab3:
            st.header("Bias Circuit Discovery")
            st.markdown(f"""
            Finding neurons that encode **{protected_attr}** by comparing activations
            between **{value_a}** and **{value_b}** samples.
            """)

            if st.button("Discover Bias Circuit"):
                mask_a = df[protected_attr] == value_a
                mask_b = df[protected_attr] == value_b

                if mask_a.sum() > 0 and mask_b.sum() > 0:
                    # Take representative samples
                    idx_a = mask_a.values.nonzero()[0][0]
                    idx_b = mask_b.values.nonzero()[0][0]

                    x_a = X[idx_a:idx_a+1].to(device)
                    x_b = X[idx_b:idx_b+1].to(device)

                    circuit = discover_circuits(model, x_a, x_b, layer_names, top_k=5)

                    st.write(f"**Total Circuit Importance:** {circuit.total_importance:.4f}")
                    st.write("**Identified Bias Neurons:**")

                    for neuron in circuit.neurons[:15]:
                        st.write(
                            f"- **{neuron.layer_name}** Neuron {neuron.neuron_index}: "
                            f"importance = {neuron.importance_score:.4f}"
                        )

                    st.session_state['bias_circuit'] = circuit
                    st.session_state['bias_attr'] = protected_attr
                    st.success("Circuit saved! Go to 'Ablation & Pruning' tab.")
                else:
                    st.error("Not enough samples in one of the groups.")

        # ========== Tab 4: Ablation & Pruning ==========
        with tab4:
            st.header("Ablation Study & Safety Pruning")

            if 'bias_circuit' not in st.session_state:
                st.warning("Please discover the bias circuit first (Tab 3).")
            else:
                circuit = st.session_state['bias_circuit']

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
                        default=layer_neurons[ablate_layer][:3]
                    )

                    if neurons_to_ablate and st.button("Run Ablation"):
                        mask_a = df[protected_attr] == value_a
                        mask_b = df[protected_attr] == value_b

                        # Get predictions before and after ablation
                        results = []

                        for idx in range(min(200, len(df))):
                            x = X[idx:idx+1].to(device)
                            group = df.iloc[idx][protected_attr]

                            with torch.no_grad():
                                orig_pred = model(x).softmax(dim=1)[0, 2].item()  # P(Good)

                            ablated_out = run_ablation(model, x, ablate_layer, neurons_to_ablate)
                            ablated_pred = ablated_out.softmax(dim=1)[0, 2].item()

                            results.append({
                                'Group': group,
                                'Original': orig_pred,
                                'Ablated': ablated_pred,
                                'Diff': ablated_pred - orig_pred
                            })

                        results_df = pd.DataFrame(results)

                        # Show scatter plot
                        fig = px.scatter(
                            results_df,
                            x='Original',
                            y='Ablated',
                            color='Group',
                            title="Original vs Ablated P(Good)"
                        )
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines', name='No Change',
                            line=dict(dash='dash')
                        ))
                        st.plotly_chart(fig, width='stretch')

                        # Show effect by group
                        st.subheader("Average Effect by Group")
                        effect_by_group = results_df.groupby('Group')['Diff'].mean()
                        st.bar_chart(effect_by_group)

                # Safety Pruning
                st.subheader("Safety Pruning")
                prune_layer = st.selectbox("Layer to prune", list(layer_neurons.keys()), key="prune_layer")

                if prune_layer:
                    prune_neurons = st.multiselect(
                        "Neurons to prune permanently",
                        layer_neurons[prune_layer],
                        default=layer_neurons[prune_layer][:2],
                        key="prune_neurons"
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
                        df_with_preds['Orig_Pred'] = orig_preds.numpy()
                        df_with_preds['Pruned_Pred'] = pruned_preds.numpy()

                        orig_good_a = (df_with_preds[df_with_preds[protected_attr] == value_a]['Orig_Pred'] == 2).mean()
                        orig_good_b = (df_with_preds[df_with_preds[protected_attr] == value_b]['Orig_Pred'] == 2).mean()
                        pruned_good_a = (df_with_preds[df_with_preds[protected_attr] == value_a]['Pruned_Pred'] == 2).mean()
                        pruned_good_b = (df_with_preds[df_with_preds[protected_attr] == value_b]['Pruned_Pred'] == 2).mean()

                        orig_gap = abs(orig_good_a - orig_good_b)
                        pruned_gap = abs(pruned_good_a - pruned_good_b)

                        st.subheader("Fairness Impact")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Bias Gap", f"{orig_gap:.2%}")
                        with col2:
                            st.metric("Pruned Bias Gap", f"{pruned_gap:.2%}",
                                     delta=f"{pruned_gap - orig_gap:.2%}",
                                     delta_color="inverse")

    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.info("Please ensure the Credit Score dataset is in data/raw/CreditScore/")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
