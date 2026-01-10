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
from src.interpretability.hooks import get_all_activations


def activations_tab(
    model, input_tensor, activations, layer_names, layer_prefix="layer"
):
    st.header("Inside the Black Box")
    st.info("Using generic HookManager from src.interpretability")

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        if f"{layer_prefix}1" in activations:
            l1_acts = activations[f"{layer_prefix}1"].cpu().numpy().flatten()
            st.subheader(f"Layer 1 Activations ({len(l1_acts)} Neurons)")
            fig_l1 = px.imshow(
                [l1_acts],
                labels=dict(x="Neuron Index", y="Layer", color="Activation"),
                color_continuous_scale="Viridis",
                height=200,
            )
            fig_l1.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_l1, width="stretch")
        else:
            st.write(f"No {layer_prefix}1 in activations: {list(activations.keys())}")

    with col_l2:
        if f"{layer_prefix}2" in activations:
            l2_acts = activations[f"{layer_prefix}2"].cpu().numpy().flatten()
            st.subheader(f"Layer 2 Activations ({len(l2_acts)} Neurons)")
            fig_l2 = px.imshow(
                [l2_acts],
                labels=dict(x="Neuron Index", y="Layer", color="Activation"),
                color_continuous_scale="Viridis",
                height=200,
            )
            fig_l2.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_l2, width="stretch")
        else:
            st.write(f"No {layer_prefix}2 in activations: {list(activations.keys())}")

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
