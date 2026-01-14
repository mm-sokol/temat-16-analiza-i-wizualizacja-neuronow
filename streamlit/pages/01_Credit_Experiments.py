import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
import copy
import sys
from pathlib import Path

# ==========================================
# PATH CONFIGURATION
# ==========================================
current_file = Path(__file__).resolve()
streamlit_dir = current_file.parent.parent  # streamlit/
project_root = streamlit_dir.parent  # project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(streamlit_dir))  # For data_gen, model_utils

# ==========================================
# IMPORTS
# ==========================================
from data_gen import generate_synthetic_data
from model_utils import train_model, SimpleMLP
from src.interpretability import HookManager, discover_circuits
from src.interpretability.hooks import get_all_activations
from src.interpretability.pruning import prune_biased_neurons
from src.data.credit_score_data import (
    load_credit_score_dataset,
    get_protected_attribute_values,
    PROTECTED_ATTRIBUTES,
)
from src.modeling.train import train_real_credit_model

st.set_page_config(page_title="Credit Experiments", page_icon="💰", layout="wide")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_synthetic_lab():
    st.header("1. Synthetic Data Lab (Tutorial)")
    st.markdown("""
    **Goal:** Understand the mechanics of bias injection and removal in a controlled environment.
    Here, 'Zip Code' is explicitly correlated with the target.
    """)

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Dataset Size", 500, 2000, 1000)
    with col2:
        seed = st.number_input("Seed", 42)

    @st.cache_data
    def get_synth_data(n, s):
        return generate_synthetic_data(n, s)

    @st.cache_resource
    def get_synth_model(_df):
        return train_model(_df, epochs=150, device=device)

    df = get_synth_data(n_samples, seed)
    model = get_synth_model(df)

    tab1, tab2, tab3 = st.tabs(["🔍 Inspection", "🧠 Circuit Discovery", "✂️ Safety Pruning"])

    with tab1:
        st.subheader("Single Applicant Inspection")
        idx = st.slider("Select Applicant", 0, len(df)-1, 0)
        sample = df.iloc[idx]
        
        feat = torch.tensor(
            sample[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']].values, 
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(feat).item()
            
        c1, c2 = st.columns(2)
        with c1: 
            st.dataframe(sample)
        with c2:
            st.metric("Model Score", f"{score:.4f}")
            st.metric("Zip Code", int(sample['Zip Code']))

        st.caption("Internal Activations:")
        manager = HookManager(model)
        acts = get_all_activations(model, feat, [l for l in manager.list_layers() if 'layer' in l])
        if 'layer1' in acts:
            st.plotly_chart(px.bar(y=acts['layer1'].cpu().flatten(), title="Layer 1 Activations"), use_container_width=True)

    with tab2:
        st.subheader("Bias Circuit Discovery")
        st.info("We flip the 'Zip Code' and see which neurons react.")
        
        if st.button("Run Auto-Discovery"):
            feat = torch.tensor(df[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']].values, dtype=torch.float32).to(device)
            
            cf_feat = feat.clone()
            cf_feat[:, 3] = 1 - cf_feat[:, 3]
            
            manager = HookManager(model)
            layers = [l for l in manager.list_layers() if 'layer' in l]
            
            circuit = discover_circuits(model, feat[:100], cf_feat[:100], layers, top_k=5)
            
            st.session_state['synth_circuit'] = circuit
            st.success(f"Found {len(circuit.neurons)} critical neurons.")
            for n in circuit.neurons:
                st.write(f"- {n.layer_name}, Neuron {n.neuron_index} (Imp: {n.importance_score:.3f})")

    with tab3:
        st.subheader("Safety Pruning")
        
        if 'synth_circuit' not in st.session_state:
            st.warning("Run Circuit Discovery in Tab 2 first.")
        else:
            circuit = st.session_state['synth_circuit']
            candidates = [f"{n.layer_name}:{n.neuron_index}" for n in circuit.neurons]
            
            to_prune = st.multiselect("Select neurons to remove", candidates, default=candidates[:1])
            
            if st.button("Apply Pruning & Evaluate"):
                pruned_model = copy.deepcopy(model)
                
                for item in to_prune:
                    layer, idx = item.split(":")
                    pruned_model, _ = prune_biased_neurons(pruned_model, layer, [int(idx)], copy=False)
                
                preds_orig = []
                preds_pruned = []
                
                X = torch.tensor(df[['Income', 'Credit History', 'Age', 'Zip Code', 'Random Noise']].values, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    preds_orig = model(X).cpu().flatten().numpy()
                    preds_pruned = pruned_model(X).cpu().flatten().numpy()
                
                df_res = df.copy()
                df_res['Orig'] = preds_orig
                df_res['Pruned'] = preds_pruned
                
                gap_orig = abs(df_res[df_res['Zip Code']==1]['Orig'].mean() - df_res[df_res['Zip Code']==0]['Orig'].mean())
                gap_pruned = abs(df_res[df_res['Zip Code']==1]['Pruned'].mean() - df_res[df_res['Zip Code']==0]['Pruned'].mean())
                
                orig_acc = ((df_res['Orig'] > 0.5) == df_res['Target']).mean()
                pruned_acc = ((df_res['Pruned'] > 0.5) == df_res['Target']).mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Original Bias Gap", f"{gap_orig:.3f}")
                c2.metric("Pruned Bias Gap", f"{gap_pruned:.3f}", delta=f"{gap_pruned-gap_orig:.3f}", delta_color="inverse")
                c3.metric("Accuracy Change", f"{pruned_acc:.1%}", delta=f"{pruned_acc-orig_acc:.1%}")
                
                st.plotly_chart(px.scatter(df_res, x="Orig", y="Pruned", color="Zip Code", title="Impact Analysis"), use_container_width=True)


def run_real_lab():
    st.header("2. Real Data Lab (Kaggle)")
    st.markdown("""
    **Goal:** Detect and mitigate bias in a model trained on complex, real-world data.
    Note: Real bias is often distributed across many neurons (redundancy). 
    **You may need to prune aggressively.**
    """)

    col1, col2 = st.columns(2)
    with col1:
        max_samples = st.slider("Sample Count", 2000, 10000, 5000)
    with col2:
        seed = st.number_input("Seed", 42, key='real_seed')

    @st.cache_data
    def get_real_data(m, s):
        try:
            return load_credit_score_dataset(max_samples=m, random_state=s)
        except FileNotFoundError:
            return None

    data_res = get_real_data(max_samples, seed)
    if not data_res:
        st.error("Missing data/raw/CreditScore/train.csv")
        st.stop()
        
    df, X, y, encodings, info = data_res

    @st.cache_resource
    def get_real_model(_X, _y):
        return train_real_credit_model(
            _X, _y, _X.shape[1], device, epochs=30
        )

    with st.spinner("Training Real Model..."):
        model = get_real_model(X, y)

    st.subheader("Bias Configuration")
    
    safe_index = 0
    if len(PROTECTED_ATTRIBUTES) > 1:
        safe_index = 1
    
    prot_attr = st.selectbox(
        "Protected Attribute", 
        PROTECTED_ATTRIBUTES, 
        index=safe_index,
        help=f"Loaded attributes: {PROTECTED_ATTRIBUTES}" # Pokaże w dymku co się załadowało
    )
    # --- FIX END ---
    
    vals = get_protected_attribute_values(df, prot_attr)
    val_a = vals[0]
    val_b = vals[1] if len(vals) > 1 else vals[0]
    
    st.write(f"Comparing: **{val_a}** vs **{val_b}**")

    t_detect, t_prune = st.tabs(["🕵️ Detect Bias", "✂️ Mitigate (Pruning)"])

    with t_detect:
        if st.button("Run Detection Analysis"):
            with torch.no_grad():
                preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
            
            df['Pred'] = preds
            r_a = (df[df[prot_attr]==val_a]['Pred'] == 2).mean()
            r_b = (df[df[prot_attr]==val_b]['Pred'] == 2).mean()
            
            c1, c2 = st.columns(2)
            c1.metric(f"Approval Rate ({val_a})", f"{r_a:.1%}")
            c2.metric(f"Approval Rate ({val_b})", f"{r_b:.1%}", delta=f"Gap: {r_a-r_b:.1%}")
            
            st.write("Running Circuit Discovery...")
            mask_a = df[prot_attr] == val_a
            mask_b = df[prot_attr] == val_b
            
            if mask_a.sum() == 0 or mask_b.sum() == 0:
                st.error("Not enough samples for selected attribute values.")
                st.stop()
                
            idx_a = mask_a.values.nonzero()[0][0]
            idx_b = mask_b.values.nonzero()[0][0]
            
            manager = HookManager(model)
            layers = [n for n, _ in model.named_modules() if isinstance(_, torch.nn.Linear)]
            
            circuit = discover_circuits(
                model, 
                X[idx_a:idx_a+1].to(device), 
                X[idx_b:idx_b+1].to(device), 
                layers, top_k=20 
            )
            st.session_state['real_circuit'] = circuit
            st.success(f"Discovered {len(circuit.neurons)} potential bias neurons.")

    with t_prune:
        if 'real_circuit' not in st.session_state:
            st.warning("Run Detection first.")
        else:
            circuit = st.session_state['real_circuit']
            
            st.markdown("### Aggressive Pruning Strategy")
            st.info("In real models, bias is distributed. You need to prune multiple neurons to see an effect.")
            
            prune_count = st.slider("Aggressiveness (Number of Neurons to Prune)", 1, 20, 5)
            
            if st.button("Apply Aggressive Pruning"):
                pruned_model = copy.deepcopy(model)
                targets = circuit.neurons[:prune_count]
                
                for n in targets:
                    pruned_model, _ = prune_biased_neurons(pruned_model, n.layer_name, [n.neuron_index], copy=False)
                
                with torch.no_grad():
                    p_orig = model(X.to(device)).argmax(dim=1).cpu().numpy()
                    p_pruned = pruned_model(X.to(device)).argmax(dim=1).cpu().numpy()
                
                df['Pred_Orig'] = p_orig
                df['Pred_Pruned'] = p_pruned
                
                ra_o = (df[df[prot_attr]==val_a]['Pred_Orig'] == 2).mean()
                rb_o = (df[df[prot_attr]==val_b]['Pred_Orig'] == 2).mean()
                gap_orig = abs(ra_o - rb_o)
                
                ra_p = (df[df[prot_attr]==val_a]['Pred_Pruned'] == 2).mean()
                rb_p = (df[df[prot_attr]==val_b]['Pred_Pruned'] == 2).mean()
                gap_pruned = abs(ra_p - rb_p)
                
                acc_o = (df['Pred_Orig'] == y.numpy()).mean()
                acc_p = (df['Pred_Pruned'] == y.numpy()).mean()
                
                st.divider()
                m1, m2 = st.columns(2)
                m1.metric("Bias Gap", f"{gap_pruned:.1%}", f"{gap_pruned - gap_orig:.1%}", delta_color="inverse")
                m2.metric("Model Accuracy", f"{acc_p:.1%}", f"{acc_p - acc_o:.1%}")
                
                if gap_pruned < gap_orig:
                    st.success("SUCCESS: Bias reduced!")
                else:
                    st.warning("Bias didn't decrease. Try increasing aggressiveness.")


st.sidebar.title("Laboratory Selection")
lab_mode = st.sidebar.radio("Choose Dataset", ["Synthetic (Tutorial)", "Real (Kaggle)"])

if lab_mode == "Synthetic (Tutorial)":
    run_synthetic_lab()
else:
    run_real_lab()