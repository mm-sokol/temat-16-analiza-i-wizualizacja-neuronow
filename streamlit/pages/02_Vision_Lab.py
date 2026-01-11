import streamlit as st
import torch
import torch.nn.functional as F
import plotly.express as px
import numpy as np
import sys
from pathlib import Path

# Captum Imports (Wymaga: uv add captum)
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

# ==========================================
# PATH CONFIGURATION
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# ==========================================
# IMPORTS
# ==========================================
from src.modeling.architecture import SimpleCNN, InterpretableMLP
from src.dataset import load_mnist
from src.plots import visualize_activations
from src.interpretability.hooks import HookManager, get_all_activations

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Vision Lab",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Vision Lab: Architecture Comparison")
st.markdown("""
**Mechanistic Interpretability on Image Data.**
Compare how a **Convolutional Neural Network (CNN)** preserves spatial structure versus how a **Multi-Layer Perceptron (MLP)** flattens it.
""")

# ==========================================
# SIDEBAR & CONFIGURATION
# ==========================================
st.sidebar.header("Experiment Settings")

model_type = st.sidebar.radio(
    "Select Architecture",
    ["Convolutional (CNN)", "Standard (MLP)"],
    help="CNN uses spatial filters. MLP flattens input to a vector."
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.caption(f"Device: {device.upper()}")

if not CAPTUM_AVAILABLE:
    st.sidebar.error("Captum not installed! Run `uv add captum`")

# ==========================================
# DATA LOADING (Cached)
# ==========================================
@st.cache_resource
def get_mnist_batch():
    """Loads a single batch of MNIST test data for inspection."""
    _, test_loader = load_mnist(batch_size=32)
    images, labels = next(iter(test_loader))
    return images, labels

try:
    images, labels = get_mnist_batch()
    st.sidebar.success("MNIST Data Loaded")
except Exception as e:
    st.error(f"Failed to load MNIST: {e}")
    st.stop()

# ==========================================
# MODEL LOADING & TRAINING (Cached)
# ==========================================
@st.cache_resource
def get_trained_model(arch_type, _device):
    if arch_type == "Convolutional (CNN)":
        model = SimpleCNN().to(_device)
    else:
        model = InterpretableMLP(784, 128, 10).to(_device)
    
    # Quick Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader, _ = load_mnist(batch_size=64)
    
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(_device), target.to(_device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if i > 50: break 
        
    model.eval()
    return model

with st.spinner(f"Initializing and optimizing {model_type}..."):
    model = get_trained_model(model_type, device)

# ==========================================
# SAMPLE SELECTION
# ==========================================
st.sidebar.header("Input Selection")
sample_idx = st.sidebar.slider("Select Test Image", 0, len(images)-1, 0)

img_tensor = images[sample_idx:sample_idx+1].to(device)
true_label = labels[sample_idx].item()

# Forward pass
with torch.no_grad():
    output = model(img_tensor)
    pred_label = output.argmax(dim=1).item()
    probs = F.softmax(output, dim=1).cpu().numpy().flatten()

# ==========================================
# MAIN UI LAYOUT
# ==========================================
col_left, col_mid, col_right = st.columns([1, 1, 2])

# --- COLUMN 1: INPUT ---
with col_left:
    st.subheader("Input")
    img_display = images[sample_idx].squeeze().numpy()
    fig_img = px.imshow(img_display, color_continuous_scale='gray', title=f"Label: {true_label}")
    fig_img.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
    fig_img.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    st.plotly_chart(fig_img, use_container_width=True)

# --- COLUMN 2: OUTPUT ---
with col_mid:
    st.subheader("Prediction")
    color = "green" if pred_label == true_label else "red"
    st.markdown(f"### :{color}[{pred_label}]")
    
    fig_probs = px.bar(
        x=list(range(10)), y=probs, labels={'x': 'Digit', 'y': 'Probability'}, title="Confidence"
    )
    fig_probs.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_probs, use_container_width=True)

# --- COLUMN 3: EXPLAINABILITY ---
with col_right:
    st.subheader("Explainability Tools")
    
    tab1, tab2 = st.tabs(["🔍 Internal Activations", "✨ Pixel Attribution (Captum)"])
    
    # TAB 1: MECHANISTIC (What happened inside?)
    with tab1:
        st.caption("See which neurons fired strongly.")
        manager = HookManager(model)
        display_layers = [l for l in manager.list_layers() if any(x in l for x in ['conv', 'fc', 'layer'])]
        
        target_layer = st.selectbox("Select Layer", display_layers, index=0 if display_layers else None)
        
        if target_layer:
            activations = get_all_activations(model, img_tensor, [target_layer])
            if target_layer in activations:
                visualize_activations(activations[target_layer], target_layer)
    
    # TAB 2: ATTRIBUTION (Why this decision?)
    with tab2:
        st.caption("See which pixels contributed most to the prediction.")
        
        if CAPTUM_AVAILABLE:
            if st.button("Run Integrated Gradients"):
                ig = IntegratedGradients(model)
                
                # Captum needs requires_grad=True
                input_var = img_tensor.clone().detach().requires_grad_(True)
                
                # Compute attributions against target class
                attributions, delta = ig.attribute(
                    input_var, 
                    target=pred_label, 
                    return_convergence_delta=True
                )
                
                # Visualization
                attr_np = attributions.squeeze().detach().cpu().numpy()
                
                # Handle MLP flattening for visualization if needed
                if len(attr_np.shape) == 1: # MLP Output
                    attr_np = attr_np.reshape(28, 28)
                
                # Heatmap
                fig_attr = px.imshow(
                    attr_np, 
                    color_continuous_scale='RdBu_r', # Red = Positive contribution, Blue = Negative
                    origin='upper',
                    title=f"Why did model choose {pred_label}?"
                )
                st.plotly_chart(fig_attr, use_container_width=True)
                
                st.info("Red pixels increased the probability of this class. Blue pixels decreased it.")
        else:
            st.warning("Please install captum to use this feature.")