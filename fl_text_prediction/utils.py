# =============================================================
# utils.py — Data Loading, Partitioning, Metrics & Plotting
# =============================================================
# This module is the backbone of the project.  It handles:
#   1. Loading the IMDB dataset via tensorflow_datasets.
#   2. Building a shared vocabulary / tokenizer.
#   3. Non-IID partitioning across NUM_CLIENTS clients.
#   4. Helper functions for perplexity and metrics tables.
#   5. Matplotlib plotting utilities.
#   6. A text-inference helper used for the demo.
#
# NON-IID PARTITIONING RATIONALE
# ───────────────────────────────
# Real keyboard/email data is highly non-IID: one user writes
# formal English, another uses slang, a third sends short bursts.
# We simulate this by:
#   • Assigning IMDB reviews by sentiment (pos/neg) to different clients.
#   • Further splitting within each sentiment class so clients see
#     different review *subsets*, giving each a unique word distribution.
# =============================================================

import os
import math
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works in Colab too)
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import VOCAB_SIZE, SEQ_LEN, BATCH_SIZE

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
NUM_CLIENTS   = 10     # total simulated devices
TOKENIZER_PATH = "tokenizer.pkl"   # cached so all scripts share it


# ==================================================================
# 1.  DATA LOADING
# ==================================================================

def load_imdb_raw():
    """
    Download (once) and return raw IMDB train/test splits as lists.

    Returns
    -------
    train_texts : list[str]  — 25,000 review strings
    train_labels: list[int]  — 0=negative, 1=positive
    test_texts  : list[str]
    test_labels : list[int]
    """
    print("[utils] Loading IMDB dataset via tensorflow_datasets …")
    # as_supervised=True → (text, label) pairs
    ds_train, ds_test = tfds.load(
        "imdb_reviews",
        split=["train", "test"],
        as_supervised=True,
        shuffle_files=False,        # deterministic for reproducibility
    )

    train_texts, train_labels = [], []
    for text, label in tfds.as_numpy(ds_train):
        train_texts.append(text.decode("utf-8"))
        train_labels.append(int(label))

    test_texts, test_labels = [], []
    for text, label in tfds.as_numpy(ds_test):
        test_texts.append(text.decode("utf-8"))
        test_labels.append(int(label))

    print(f"[utils] Loaded {len(train_texts)} train + {len(test_texts)} test reviews.")
    return train_texts, train_labels, test_texts, test_labels


# ==================================================================
# 2.  TOKENIZER / VOCABULARY
# ==================================================================

def build_or_load_tokenizer(texts=None):
    """
    Build a Keras Tokenizer on `texts` (if not cached) or load from disk.

    The tokenizer is shared globally: the server fits it once on ALL
    text data, then distributes the vocabulary to clients.  This
    mirrors a real-world scenario where the vocabulary is pre-built
    and baked into the keyboard app.

    Parameters
    ----------
    texts : list[str] | None — required on first call; ignored if cached

    Returns
    -------
    tokenizer : fitted keras Tokenizer
    """
    if os.path.exists(TOKENIZER_PATH):
        print(f"[utils] Loading cached tokenizer from '{TOKENIZER_PATH}' …")
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)

    if texts is None:
        raise ValueError("texts must be provided to build the tokenizer.")

    print(f"[utils] Building tokenizer (vocab_size={VOCAB_SIZE}) …")
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<OOV>",   # index 1 — out-of-vocabulary words map here
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
    )
    tokenizer.fit_on_texts(texts)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[utils] Tokenizer saved to '{TOKENIZER_PATH}'.")
    return tokenizer


def texts_to_sequences(tokenizer, texts, seq_len=SEQ_LEN):
    """
    Convert raw text strings → padded integer sequences suitable for
    the LSTM.  Each token at position t becomes the target for the
    subsequence ending at t-1 (next-word prediction setup).

    Parameters
    ----------
    tokenizer : fitted Keras Tokenizer
    texts     : list[str]
    seq_len   : fixed window length

    Returns
    -------
    X : np.ndarray, shape (N, seq_len)   — input windows
    y : np.ndarray, shape (N,)           — target tokens (next word)
    """
    X_list, y_list = [], []

    for text in texts:
        # Convert text → list of integer token IDs
        token_ids = tokenizer.texts_to_sequences([text])[0]

        # Slide a window of (seq_len + 1) tokens across the review.
        # Input: ids[i : i+seq_len]   Target: ids[i+seq_len]
        for i in range(len(token_ids) - seq_len):
            X_list.append(token_ids[i : i + seq_len])
            y_list.append(token_ids[i + seq_len])

    if not X_list:
        # Edge case: text too short — return empty arrays
        return np.zeros((0, seq_len), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    X = pad_sequences(X_list, maxlen=seq_len, padding="pre", truncating="pre")
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ==================================================================
# 3.  NON-IID DATA PARTITIONING
# ==================================================================

def partition_non_iid(train_texts, train_labels, num_clients=NUM_CLIENTS):
    """
    Partition IMDB reviews across `num_clients` in a non-IID fashion.

    Strategy
    --------
    • Clients 0-4  : positive reviews (label = 1)
    • Clients 5-9  : negative reviews (label = 0)
    • Within each sentiment group, reviews are split roughly equally
      but with different subsets, ensuring distinct word distributions.

    This simulates users with different writing styles / topics.

    Parameters
    ----------
    train_texts  : list[str]
    train_labels : list[int]
    num_clients  : int

    Returns
    -------
    client_data : dict[int → (texts: list[str], labels: list[int])]
    """
    print(f"[utils] Partitioning data across {num_clients} clients (non-IID) …")

    # Split by sentiment
    pos_texts = [t for t, l in zip(train_texts, train_labels) if l == 1]
    neg_texts = [t for t, l in zip(train_texts, train_labels) if l == 0]

    # Shuffle each group deterministically
    rng = np.random.default_rng(seed=42)
    rng.shuffle(pos_texts)
    rng.shuffle(neg_texts)

    half = num_clients // 2  # 5 positive clients, 5 negative clients

    client_data = {}
    for i in range(half):
        # Each positive client gets a non-overlapping slice of pos_texts
        chunk = pos_texts[i::half][:200]   # cap at 200 reviews per client         # every 5th review starting at i
        client_data[i] = (chunk, [1] * len(chunk))

    for i in range(half, num_clients):
        j = i - half
        chunk = neg_texts[j::half][:200]   # cap at 200 reviews per client         # every 5th review starting at i
        client_data[i] = (chunk, [0] * len(chunk))

    # Report distribution
    for cid, (texts, labels) in client_data.items():
        sentiment = "positive" if labels[0] == 1 else "negative"
        print(f"  Client {cid:2d}: {len(texts):5d} reviews  [{sentiment}]")

    return client_data


def prepare_client_datasets(client_data, tokenizer, seq_len=SEQ_LEN,
                             batch_size=BATCH_SIZE):
    """
    Convert each client's raw texts → tf.data.Dataset of (X, y) batches.

    Parameters
    ----------
    client_data : dict[int → (texts, labels)]
    tokenizer   : fitted Keras Tokenizer
    seq_len     : int
    batch_size  : int

    Returns
    -------
    datasets : dict[int → tf.data.Dataset]
    """
    print("[utils] Preparing client tf.data.Datasets …")
    datasets = {}
    for cid, (texts, _) in client_data.items():
        X, y = texts_to_sequences(tokenizer, texts, seq_len)
        if len(X) == 0:
            print(f"  [Warning] Client {cid} has no usable sequences — skipping.")
            continue
        ds = (tf.data.Dataset
              .from_tensor_slices((X, y))
              .shuffle(buffer_size=min(len(X), 5000), seed=cid)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))
        datasets[cid] = ds
        print(f"  Client {cid:2d}: {len(X):6d} sequences → {math.ceil(len(X)/batch_size)} batches")
    return datasets


def dataset_to_numpy(dataset: tf.data.Dataset):
    """
    Pull all batches from a tf.data.Dataset into numpy arrays.
    Used by the Flower NumPyClient for fit() / evaluate().
    """
    Xs, ys = [], []
    for xb, yb in dataset:
        Xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ==================================================================
# 4.  METRICS
# ==================================================================

def compute_perplexity(loss: float) -> float:
    """
    Perplexity = exp(cross-entropy loss).
    Lower perplexity = better language model.
    A perplexity of 10,000 = random guessing over 10k words.
    """
    return math.exp(min(loss, 100))   # clip to avoid overflow


def metrics_table(history: dict):
    """
    Pretty-print a table of per-round metrics.

    Parameters
    ----------
    history : dict with keys 'round', 'loss', 'accuracy', 'perplexity'
              (each a list of values, one per FL round)
    """
    header = f"{'Round':>6} | {'Loss':>8} | {'Accuracy':>10} | {'Perplexity':>12}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r, lo, ac, pp in zip(history["round"],
                              history["loss"],
                              history["accuracy"],
                              history["perplexity"]):
        print(f"{r:>6} | {lo:>8.4f} | {ac:>10.4f} | {pp:>12.2f}")
    print(sep)


# ==================================================================
# 5.  PLOTTING
# ==================================================================

def plot_metrics(history: dict, title_prefix: str = "Federated",
                 save_dir: str = "."):
    """
    Generate and save two PNG figures:
      1. Loss curve across FL rounds.
      2. Accuracy curve across FL rounds.

    Parameters
    ----------
    history     : dict with keys 'round', 'loss', 'accuracy'
    title_prefix: string prepended to plot titles
    save_dir    : directory to write PNG files
    """
    os.makedirs(save_dir, exist_ok=True)
    rounds = history["round"]

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, history["loss"], marker="o", color="#E74C3C", linewidth=2,
            label="Train Loss")
    if "val_loss" in history:
        ax.plot(rounds, history["val_loss"], marker="s", color="#E74C3C",
                linestyle="--", linewidth=2, label="Val Loss")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title(f"{title_prefix} — Loss per Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(save_dir, f"{title_prefix.lower().replace(' ','_')}_loss.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Saved loss plot → {loss_path}")

    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, history["accuracy"], marker="o", color="#2980B9", linewidth=2,
            label="Top-1 Accuracy")
    if "val_accuracy" in history:
        ax.plot(rounds, history["val_accuracy"], marker="s", color="#2980B9",
                linestyle="--", linewidth=2, label="Val Accuracy")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(f"{title_prefix} — Accuracy per Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = os.path.join(save_dir, f"{title_prefix.lower().replace(' ','_')}_accuracy.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Saved accuracy plot → {acc_path}")


def plot_comparison(fed_history: dict, base_history: dict, save_dir: str = "."):
    """
    Side-by-side comparison of Federated vs Centralized training.
    Saves a single PNG with two subplots.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss comparison
    axes[0].plot(fed_history["round"],  fed_history["loss"],
                 marker="o", color="#E74C3C", label="Federated")
    axes[0].plot(base_history["round"], base_history["loss"],
                 marker="s", color="#8E44AD", linestyle="--", label="Centralized")
    axes[0].set_title("Loss: Federated vs Centralized")
    axes[0].set_xlabel("Round / Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy comparison
    axes[1].plot(fed_history["round"],  fed_history["accuracy"],
                 marker="o", color="#2980B9", label="Federated")
    axes[1].plot(base_history["round"], base_history["accuracy"],
                 marker="s", color="#27AE60", linestyle="--", label="Centralized")
    axes[1].set_title("Accuracy: Federated vs Centralized")
    axes[1].set_xlabel("Round / Epoch")
    axes[1].set_ylabel("Top-1 Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Federated Learning vs Centralized Baseline", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "comparison_fed_vs_centralized.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[utils] Saved comparison plot → {path}")


def plot_client_personalization(client_results: dict, save_dir: str = "."):
    """
    Bar chart showing per-client accuracy before/after personalization.

    Parameters
    ----------
    client_results : dict[int → {'before': float, 'after': float}]
    """
    os.makedirs(save_dir, exist_ok=True)
    client_ids = sorted(client_results.keys())
    before = [client_results[c]["before"] for c in client_ids]
    after  = [client_results[c]["after"]  for c in client_ids]

    x = np.arange(len(client_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, before, width, label="Global Model",  color="#95A5A6")
    bars2 = ax.bar(x + width/2, after,  width, label="Personalized",  color="#2ECC71")

    ax.set_xlabel("Client ID")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Per-Client Accuracy: Global vs Personalized Model")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in client_ids])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars with values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(save_dir, "client_personalization.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[utils] Saved personalization chart → {path}")


# ==================================================================
# 6.  INFERENCE / DEMO
# ==================================================================

def predict_next_words(model, tokenizer, seed_text: str,
                       num_words: int = 5, seq_len: int = SEQ_LEN,
                       temperature: float = 1.0) -> str:
    """
    Predict the next `num_words` tokens given a seed phrase.

    Parameters
    ----------
    model       : trained Keras model
    tokenizer   : fitted Keras Tokenizer
    seed_text   : starting text, e.g. "the movie was really"
    num_words   : number of tokens to generate
    seq_len     : expected model input length
    temperature : sampling temperature (1.0 = greedy-ish, >1 = more random)

    Returns
    -------
    predicted_text : seed_text + " " + generated tokens
    """
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    generated = []

    current_text = seed_text.lower()
    for _ in range(num_words):
        # Tokenize current text
        token_ids = tokenizer.texts_to_sequences([current_text])[0]

        # Pad/truncate to seq_len
        padded = pad_sequences([token_ids], maxlen=seq_len, padding="pre", truncating="pre")

        # Get probability distribution (shape: (1, vocab_size))
        probs = model.predict(padded, verbose=0)[0]

        # Apply temperature scaling
        if temperature != 1.0:
            probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(probs)
            probs /= probs.sum()

        # Sample from distribution (top-5 nucleus to avoid <OOV>)
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs   = probs[top_indices]
        top_probs   /= top_probs.sum()

        chosen_idx = np.random.choice(top_indices, p=top_probs)
        chosen_word = index_to_word.get(chosen_idx, "<OOV>")

        # Skip OOV tokens
        if chosen_word == "<OOV>":
            chosen_word = index_to_word.get(top_indices[1], "the")

        generated.append(chosen_word)
        current_text += " " + chosen_word

    return seed_text + " " + " ".join(generated)


# ==================================================================
# Quick self-test
# ==================================================================
if __name__ == "__main__":
    print("=== utils.py self-test ===")
    print(f"compute_perplexity(5.0) = {compute_perplexity(5.0):.2f}")   # ≈ 148.41
    print(f"compute_perplexity(3.0) = {compute_perplexity(3.0):.2f}")   # ≈ 20.09
    print("\nLoading IMDB data …")
    tr_txt, tr_lbl, ts_txt, ts_lbl = load_imdb_raw()
    print(f"Sample review: {tr_txt[0][:80]} …")
    print("\nBuilding tokenizer …")
    tok = build_or_load_tokenizer(tr_txt + ts_txt)
    print(f"Vocab size: {len(tok.word_index)}")
    print("\nPartitioning …")
    cdata = partition_non_iid(tr_txt, tr_lbl, num_clients=10)
    print("\n✓  utils.py self-test passed.")
