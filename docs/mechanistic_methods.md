**Map of mechanistic interpretability methods**
---

# 1. Architecture-specific mechanistic methods

## A. MLPs (small / medium, ReLU or similar)

MLPs are the **most mechanistically tractable** models.

### Strongly mechanistic (core tools)

* **Neuron ablation (single & grouped)**
  * https://arxiv.org/pdf/1901.08644
  
* **Weight / path analysis**

  * Active ReLU paths
  * Contribution of specific weight chains
* **Piecewise-linear region analysis**
* **Minimal circuit extraction**
* **Edge (weight) ablation**
* **Rule extraction (if–then logic)**

### Representation-level

* **Activation clustering**
* **Linear probes**
* **PCA / ICA on hidden layers**

### Diagnostic / supportive

* Intermediate-layer gradients
* Activation statistics (dead neurons, saturation)

✅ *You can often fully explain what a small MLP is doing.*

---

## B. CNNs (2D convolutions)

CNNs shift interpretability from **neurons → channels → spatial features**.

### Strongly mechanistic

* **Channel-wise ablation**
* **Filter / kernel visualization**
* **Activation maximization (per channel)**
* **Subnetwork pruning (channel or block level)**
* **Causal interventions on feature maps**

### Representation-level

* **Linear probing on pooled features**
* **Concept alignment (e.g. nuclei, edges, texture)**
* **Layer-wise separability analysis**

### Spatially grounded (important for pathology)

* Feature map localization
* Receptive field analysis

⚠️ Pixel saliency ≠ mechanistic (unless tied to specific channels)

---

## C. 1D CNNs (signals, sequences, tabular-like)

Mechanistically closer to MLPs than 2D CNNs.

### Strongly mechanistic

* **Channel (feature map) ablation**
* **Kernel inspection (temporal patterns)**
* **Activation maximization in time**
* **Minimal temporal circuit discovery**
* **Edge ablation**

### Representation-level

* **Linear probing**
* **Frequency-domain analysis**
* **Temporal selectivity profiling**

### Especially useful

* Tracking *when* features fire
* Causal role of early vs late convolutions

---

# 2. Universally applicable mechanistic methods

*(largely architecture-independent)*

These methods work for **MLPs, CNNs, 1D CNNs, Transformers, etc.**

## Tier 1 – Core mechanistic tools (gold standard)

### 1. **Ablation (causal intervention)**

* Neurons
* Channels
* Layers
* Edges

> *If removing it breaks the behavior, it is part of the mechanism.*

---

### 2. **Minimal subnetwork / circuit extraction**

* Iterative pruning
* Performance-preserving subnetworks
* Lottery-ticket–style approaches

Works everywhere.

---

### 3. **Linear probing**

* Train linear classifiers on internal activations
* Measures *where information becomes explicit*

Architecture-agnostic and very strong.

---

### 4. **Activation statistics**

* Selectivity
* Sparsity
* Mutual information with labels

---

## Tier 2 – Semi-mechanistic but very useful

### 5. **Activation maximization**

* Input that maximally activates a unit
* Works for:

  * neurons
  * channels
  * attention heads

---

### 6. **Intermediate-layer saliency**

* Gradients w.r.t. hidden activations
* Attribution to *components*, not pixels

Much closer to mechanistic than IG.

---

### 7. **Representation geometry**

* PCA / ICA
* Cluster structure
* Manifold analysis

---

## Tier 3 – Concept-level (borderline mechanistic)

### 8. **TCAV / concept probes**

* Requires predefined concepts
* Tests causal influence of concepts

⚠️ Mechanistic only if concepts map cleanly to components.

---

# 3. What does *not* count as mechanistic (but can support it)

* Integrated Gradients
* Saliency maps (pixel-level)
* SmoothGrad
* LIME / SHAP

These are **functional explanations**, not mechanisms.

---

# 4. Compact summary table

| Method                  | MLP | CNN | 1D CNN | Universal |
| ----------------------- | --- | --- | ------ | --------- |
| Neuron ablation         | ✅   | ⚠️  | ⚠️     | ✅         |
| Channel ablation        | ❌   | ✅   | ✅      | ✅         |
| Edge ablation           | ✅   | ⚠️  | ⚠️     | ✅         |
| Activation maximization | ✅   | ✅   | ✅      | ✅         |
| Linear probing          | ✅   | ✅   | ✅      | ✅         |
| Circuit extraction      | ✅   | ⚠️  | ⚠️     | ✅         |
| Rule extraction         | ✅   | ❌   | ❌      | ❌         |
| Kernel visualization    | ❌   | ✅   | ✅      | ❌         |
| TCAV                    | ⚠️  | ⚠️  | ⚠️     | ✅         |

---

# 5. Recommended “mechanistic stack” (if you had to choose few)

For **any architecture**:

1. Ablation
2. Linear probing
3. Activation maximization
4. Subnetwork pruning

For **CNNs**:

* Channel-wise ablation
* Feature map localization
