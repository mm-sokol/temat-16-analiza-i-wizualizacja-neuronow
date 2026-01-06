import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from data_gen import generate_synthetic_data
from model_utils import train_model, get_activations, get_gradients, SimpleMLP

st.set_page_config(page_title="Redlining Circuit Detector", layout="wide")

@st.cache_data
def load_data():
    return generate_synthetic_data(n_samples=1000)

@st.cache_resource
def load_model(df):
    return train_model(df, epochs=200)

st.title("Mechanistic Interpretability: Detecting Redlining Circuits")
st.markdown("""
This dashboard visualizes how a neural network processes "Zip Code" (a proxy for redlining) 
to make credit decisions. We use **Activation Patching** to identify the specific neurons 
responsible for discrimination.
""")

# Load Data and Model
df = load_data()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.write(f"Using device: {device.upper()}")

@st.cache_resource
def load_model_on_device(df, device):
    return train_model(df, epochs=200, device=device)

model = load_model_on_device(df, device)

# Sidebar
st.sidebar.header("Applicant Selection")
sample_idx = st.sidebar.slider("Select Applicant Index", 0, len(df)-1, 0)

# Get Sample
sample = df.iloc[sample_idx]
input_features = sample[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']]
target = sample['Target']

# Prepare Tensor
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
    st.write(f"Zip Code: {int(input_features['Zip Code'])} ({'Advantaged' if input_features['Zip Code']==1 else 'Disadvantaged'})")

st.divider()

# Mechanistic Visualization
st.header("Inside the Black Box")

# 1. Activations
activations = get_activations(model, input_tensor)
l1_acts = activations['layer1'].cpu().numpy().flatten()
l2_acts = activations['layer2'].cpu().numpy().flatten()

col_l1, col_l2 = st.columns(2)

with col_l1:
    st.subheader("Layer 1 Activations (16 Neurons)")
    fig_l1 = px.imshow([l1_acts], labels=dict(x="Neuron Index", y="Layer", color="Activation"),
                       color_continuous_scale="Viridis", height=200)
    fig_l1.update_yaxes(showticklabels=False)
    st.plotly_chart(fig_l1, width="stretch") 


with col_l2:
    st.subheader("Layer 2 Activations (8 Neurons)")
    fig_l2 = px.imshow([l2_acts], labels=dict(x="Neuron Index", y="Layer", color="Activation"),
                       color_continuous_scale="Viridis", height=200)
    fig_l2.update_yaxes(showticklabels=False)
    st.plotly_chart(fig_l2, width="stretch")

st.divider()

# 2. Activation Patching (Counterfactual Analysis)
st.header("Detecting the Discrimination Circuit")
st.markdown("""
**Experiment:** What if we change *only* the Zip Code for this applicant?
We flip the Zip Code (0 -> 1 or 1 -> 0) and observe which neurons change their activation the most.
These neurons form the **Bias Circuit**.
""")

# Create Counterfactual
cf_features = input_features.copy()
cf_features['Zip Code'] = 1 - cf_features['Zip Code'] # Flip
cf_tensor = torch.tensor(cf_features.values, dtype=torch.float32).unsqueeze(0).to(device)

# Get Counterfactual Activations
cf_activations = get_activations(model, cf_tensor)
cf_l1_acts = cf_activations['layer1'].cpu().numpy().flatten()
cf_l2_acts = cf_activations['layer2'].cpu().numpy().flatten()

# Calculate Difference
diff_l1 = cf_l1_acts - l1_acts
diff_l2 = cf_l2_acts - l2_acts

# Visualize Differences
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("Layer 1 Sensitivity (Bias Neurons)")
    # Highlight neurons with high absolute difference
    fig_d1 = px.bar(x=range(len(diff_l1)), y=diff_l1, 
                    labels={'x': 'Neuron Index', 'y': 'Change in Activation'},
                    title="Change when Zip Code is Flipped")
    st.plotly_chart(fig_d1, width="stretch")

with col_d2:
    st.subheader("Layer 2 Sensitivity (Bias Neurons)")
    fig_d2 = px.bar(x=range(len(diff_l2)), y=diff_l2, 
                    labels={'x': 'Neuron Index', 'y': 'Change in Activation'},
                    title="Change when Zip Code is Flipped")
    st.plotly_chart(fig_d2, width="stretch")

# Conclusion
st.info("Neurons with large bars above are strongly coupled to the Zip Code feature. These are the candidates for pruning to improve fairness.")
