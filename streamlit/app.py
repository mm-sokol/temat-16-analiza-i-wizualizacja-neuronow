import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Neural Microscope",
    page_icon="🧠",
    layout="wide"
)

# Header
st.title("🧠 Neural Microscope")
st.subheader("Mechanistic Interpretability & Bias Detection Workbench")

st.markdown("""
### Welcome to the Laboratory

This tool allows you to dissect neural networks and understand their decision-making process.
We focus on **Mechanistic Interpretability** — reverse engineering the algorithms learned by the model.

### Available Experiments (Select from Sidebar):

#### 1. 💰 Credit Experiments
#### 2. 🔬 Vision Lab (CNN vs MLP)""")

# Footer / Info
st.divider()
st.info("👈 Select an experiment from the sidebar to begin.")