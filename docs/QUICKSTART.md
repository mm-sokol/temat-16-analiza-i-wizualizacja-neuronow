# Quick Start Guide

## 1. Install Dependencies
```bash
uv venv
source .venv/bin/activate
uv sync
```
or
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision pandas numpy streamlit plotly
```
you can also use python3 command instead

## 2. Run the Dashboard
```bash
cd path/temat-16-analiza-i-wizualizacja-neuronow
python -m streamlit run streamlit/app.py
```

## 3. Open Browser
Navigate to: **http://localhost:8501**

## 4. Select Demo Mode (Sidebar)
- **Credit Scoring (Synthetic)** - Best for learning, has intentional bias
- **Credit Score (Real Dataset)** - Real Kaggle data, select protected attribute
- **MNIST** - Digit recognition model analysis

## 5. Follow the Tabs
1. **Activations** - See what neurons fire
2. **Bias Detection** - Find discriminatory neurons
3. **Ablation** - Test causal importance
4. **Pruning** - Remove biased neurons

## Expected Results
- Bias Gap should **decrease** after pruning
- Accuracy should remain **> 95%** of original

