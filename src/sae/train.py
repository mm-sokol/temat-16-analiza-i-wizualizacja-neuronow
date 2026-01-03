"""
Code to train models
"""

from pathlib import Path
from sae.sparse_autoencoder import SparseAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, seed_everything
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)


def get_gpt2_weights(input_text, layer_num=-1):
    tokens = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        tokens = gpt2_model(**tokens, output_hidden_states=True)
    return tokens.hidden_states[layer_num].squeeze()


def analyze(
    text,
    sae_hidden_dim,
    epochs,
    save_path,
):

    gpt_weights = get_gpt2_weights(text)
    in_dim = gpt_weights.shape[-1]

    sae = SparseAutoencoder(in_dim, sae_hidden_dim, activation=nn.ReLU())
    print(sae)
    sae = train(sae, gpt_weights, epochs, learning_rate=0.001, sparsity_loss_weight=0.1)

    visualize_features(sae, sae_hidden_dim, save_path)
    encoded, _ = sae(gpt_weights)
    encoded = encoded.mean(dim=0).squeeze().cpu().detach().numpy()

    visualize_activation_strengths(encoded, sae_hidden_dim, save_path)


def visualize_features(
    autoencoder, num_features, save_path, filename="learned_features.png"
):
    weights = autoencoder.encoder.weight.data.cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(num_features):
        ax = axes[i // 4, i % 4]
        sns.heatmap(weights[i].reshape(1, -1), ax=ax, cmap="viridis", cbar=False)
        ax.set_title(f"Feature {i+1}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path / filename)
    plt.close()


def visualize_activation_strengths(
    encoded, num_features, save_path, filename="activation_strengths.png"
):
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_features), encoded[:num_features])
    plt.title("Activation Strengths of Learned Features")
    plt.xlabel("Feature Index")
    plt.ylabel("Activation Strength")
    plt.xticks(range(0, num_features, 2))  # Label every other feature for readability
    plt.tight_layout()
    plt.savefig(save_path / filename)
    plt.close()


def train(
    sae_model,
    weights,
    epochs,
    learning_rate,
    sparsity_loss_weight,
):
    loss = nn.MSELoss()
    optimizer = optim.Adam(sae_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    proggress_bar = tqdm(range(epochs), total=epochs, colour="magenta")
    for epoch in proggress_bar:
        proggress_bar.set_description(desc=f"Training", refresh=True)

        hidden, decoded = sae_model(weights)
        model_loss = loss(decoded, weights)
        sparsity_loss = torch.mean(torch.abs(hidden))
        epoch_loss = model_loss + sparsity_loss * sparsity_loss_weight

        epoch_loss.backward()
        optimizer.step()
        scheduler.step()

        proggress_bar.set_postfix(
            ordered_dict={"Epoch": epoch + 1, "Loss": epoch_loss.item()},
            refresh=True,
        )

    return sae_model


if __name__ == "__main__":

    seed_everything(101, True)

    text = "Nothing that was previously said remotely resembled a sensible statement. We needed further explanation."

    dest_path = FIGURES_DIR / "SAE" / datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    dest_path.mkdir(parents=True, exist_ok=True)

    analyze(text, 16, 1000, dest_path)
