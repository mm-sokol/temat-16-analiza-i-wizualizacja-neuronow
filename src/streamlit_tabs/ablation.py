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

from src.interpretability import run_ablation


def get_nerurons_to_ablate(layer_names, activations):
    selected_layer = st.selectbox(
        "Select layer to ablate", layer_names, key="ablation_layer"
    )

    if selected_layer and selected_layer in activations:
        n_neurons = activations[selected_layer].flatten().shape[0]

        neurons_to_ablate = st.multiselect(
            f"Select neurons to ablate (0-{n_neurons - 1})",
            list(range(n_neurons)),
            default=[],
            key="ablation_neurons",
        )
    return neurons_to_ablate, selected_layer


def ablation_tab(model, df, activations, layer_names):
    st.header("Ablation Study")
    st.info("Using run_ablation from src.interpretability.ablation")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    neurons_to_ablate, selected_layer = get_nerurons_to_ablate(layer_names, activations)

    if neurons_to_ablate and st.button("Run Ablation Study"):
        # Compare predictions across dataset
        original_probs = []
        ablated_probs = []
        progress = st.progress(0)
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
        comparison_df["Difference"] = (
            comparison_df["Ablated Prob"] - comparison_df["Original Prob"]
        )
        st.subheader("Ablation Results")
        # Scatter plot
        fig = px.scatter(
            comparison_df,
            x="Original Prob",
            y="Ablated Prob",
            color="Zip Code",
            hover_data=["Income", "Credit History"],
            title="Original vs Ablated Predictions",
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
        st.subheader("Average Effect by Zip Code")
        avg_diff_by_zip = comparison_df.groupby("Zip Code")["Difference"].mean()
        st.bar_chart(avg_diff_by_zip)


def get_abltaion_results(model, loader, device, selected_layer, neurons_to_ablate):

    original_probs = []
    ablated_probs = []
    targets = []
    progress = st.progress(0)
    for idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        print(idx, images.shape)
        with torch.no_grad():
            print("Here")
            orig_prob = model(images)
            orig_prob = torch.softmax(orig_prob, dim=1).squeeze()
            print("Here1", orig_prob.shape)
        ablated_output = run_ablation(model, images, selected_layer, neurons_to_ablate)
        print("Here2")
        abl_prob = ablated_output
        abl_prob = torch.softmax(abl_prob, dim=1).squeeze()
        original_probs.append(orig_prob)
        ablated_probs.append(abl_prob)
        targets.append(labels)
        progress.progress((idx + 1) / len(loader))

    all_original_probs = torch.cat(original_probs, dim=0)
    all_ablated_probs = torch.cat(ablated_probs, dim=0)
    all_labels = torch.cat(targets, dim=0)

    return all_original_probs, all_ablated_probs, all_labels


def ablation_on_images_tab(model, test_loader, activations, layer_names):
    st.header("Ablation Study")
    st.info("Using run_ablation from src.interpretability.ablation")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    neurons_to_ablate, selected_layer = get_nerurons_to_ablate(layer_names, activations)

    if "ablation_results" not in st.session_state:
        st.session_state.ablation_results = None

    if neurons_to_ablate and st.button("Run Ablation Study"):
        st.session_state.ablation_results = get_abltaion_results(
            model, test_loader, device, selected_layer, neurons_to_ablate
        )

    if st.session_state.ablation_results is not None:
        all_original_probs, all_ablated_probs, all_labels = (
            st.session_state.ablation_results
        )

        st.subheader("Ablation Results")

        choice = st.radio(
            "Select a number",
            options=list(range(1, all_original_probs.shape[1] + 1)),
            horizontal=True,
            key="ablation_choice",
        )

        fig = px.scatter(
            x=all_original_probs[:, choice - 1].numpy(),
            y=all_ablated_probs[:, choice - 1].numpy(),
            color=all_labels.numpy(),
            title="Original vs Ablated Predictions",
        )
        fig.update_layout(
            xaxis_title="Original probability",
            yaxis_title="Ablated probability",
        )
        fig.update_traces(
            hovertemplate=(
                "Original: %{x:.3f}<br>"
                "Ablated: %{y:.3f}<br>"
                "Label: %{marker.color}<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="No Change",
                line=dict(dash="dash", color="lightgray"),
            )
        )

        st.plotly_chart(fig, use_container_width=True)
