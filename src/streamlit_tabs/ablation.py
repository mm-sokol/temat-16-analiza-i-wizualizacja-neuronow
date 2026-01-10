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


def ablation_on_images_tab(model, test_loader, activations, layer_names):
    st.header("Ablation Study")
    st.info("Using run_ablation from src.interpretability.ablation")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    neurons_to_ablate, selected_layer = get_nerurons_to_ablate(layer_names, activations)

    if neurons_to_ablate and st.button("Run Ablation Study"):

        original_probs = []
        ablated_probs = []
        progress = st.progress(0)

        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                orig_prob = model(images).item()

            ablated_output = run_ablation(
                model, images, selected_layer, neurons_to_ablate
            )
            abl_prob = ablated_output.item()

            original_probs.append(orig_prob)
            ablated_probs.append(abl_prob)
            progress.progress((idx + 1) / len(test_loader))
