# =============================================================
# baseline.py — Centralised Training Baseline
# =============================================================
# PURPOSE
# ───────
# Train the SAME LSTM architecture on ALL IMDB training data in
# one shot (no federation, no privacy).  This gives us an upper-
# bound reference to compare against federated training.
#
# WHY A BASELINE?
# ───────────────
# In practice, centralized training sees the full dataset and
# benefits from perfect data mixing — it should outperform FL.
# However, federated learning has a crucial advantage: raw data
# NEVER leaves the device.  The baseline shows *how much* accuracy
# we sacrifice for privacy (the "federation gap").
#
# OUTPUT
# ──────
#   saved_models/centralized_model.keras
#   baseline_history.json  (consumed by server.py for comparison plot)
#   plots/centralized_loss.png
#   plots/centralized_accuracy.png
# =============================================================

import os
import json
import numpy as np
import tensorflow as tf

from model import build_model, VOCAB_SIZE, SEQ_LEN, BATCH_SIZE
from utils  import (
    load_imdb_raw,
    build_or_load_tokenizer,
    texts_to_sequences,
    compute_perplexity,
    metrics_table,
    plot_metrics,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
EPOCHS       = 3          # match FL rounds for fair comparison
MODELS_DIR   = "saved_models"
PLOTS_DIR    = "plots"
HISTORY_FILE = "baseline_history.json"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


def train_centralized():
    """
    Full centralized training loop.

    Steps
    ─────
    1. Load all IMDB training text.
    2. Build (or load cached) tokenizer.
    3. Convert all text → padded sequences.
    4. Train LSTM for EPOCHS epochs.
    5. Save model + metrics.
    """

    # ── 1. Load data ───────────────────────────────────────────────
    print("=" * 60)
    print("  Centralised Baseline Training")
    print("=" * 60)
    print("\n[Baseline] Loading IMDB dataset …")
    tr_txt, tr_lbl, ts_txt, ts_lbl = load_imdb_raw()

    # ── 2. Tokenizer ───────────────────────────────────────────────
    print("[Baseline] Building/loading tokenizer …")
    tokenizer = build_or_load_tokenizer(tr_txt + ts_txt)

    # ── 3. Sequence preparation ────────────────────────────────────
    print("[Baseline] Converting all train texts → sequences …")
    X_train, y_train = texts_to_sequences(tokenizer, tr_txt[:2000])
    print(f"[Baseline] Train sequences : {len(X_train):,}")

    print("[Baseline] Converting test texts → validation sequences …")
    X_val, y_val = texts_to_sequences(tokenizer, ts_txt[:300])
    print(f"[Baseline] Val sequences   : {len(X_val):,}")

    # ── 4. Build model ─────────────────────────────────────────────
    print("\n[Baseline] Building model …")
    model = build_model()
    model.summary()

    # ── 5. Callbacks ───────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1,
        ),
    ]

    # ── 6. Train ───────────────────────────────────────────────────
    print(f"\n[Baseline] Training for up to {EPOCHS} epochs …")
    keras_hist = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # ── 7. Convert keras history → our format ─────────────────────
    n_epochs = len(keras_hist.history["loss"])
    history  = {
        "round":       list(range(1, n_epochs + 1)),
        "loss":        [float(v) for v in keras_hist.history["loss"]],
        "accuracy":    [float(v) for v in keras_hist.history["accuracy"]],
        "perplexity":  [compute_perplexity(v) for v in keras_hist.history["loss"]],
        "val_loss":    [float(v) for v in keras_hist.history.get("val_loss",    [])],
        "val_accuracy":[float(v) for v in keras_hist.history.get("val_accuracy",[])],
    }

    # ── 8. Metrics table ───────────────────────────────────────────
    print("\n[Baseline] Training Metrics:")
    metrics_table(history)

    # Final validation metrics
    print("\n[Baseline] Final evaluation on validation set:")
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    val_ppl = compute_perplexity(val_loss)
    print(f"  val_loss     = {val_loss:.4f}")
    print(f"  val_accuracy = {val_acc:.4f}")
    print(f"  val_ppl      = {val_ppl:.2f}")

    # ── 9. Save model ──────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "centralized_model.keras")
    model.save(model_path)
    print(f"\n[Baseline] Model saved → {model_path}")

    # ── 10. Save history (for server.py comparison) ────────────────
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Baseline] History saved → {HISTORY_FILE}")

    # ── 11. Plots ──────────────────────────────────────────────────
    plot_metrics(history, title_prefix="Centralized", save_dir=PLOTS_DIR)

    # ── 12. Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CENTRALIZED BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Epochs trained        : {n_epochs}")
    print(f"  Final train loss      : {history['loss'][-1]:.4f}")
    print(f"  Final train accuracy  : {history['accuracy'][-1]:.4f}")
    print(f"  Final train perplexity: {history['perplexity'][-1]:.2f}")
    print(f"  Final val  loss       : {val_loss:.4f}")
    print(f"  Final val  accuracy   : {val_acc:.4f}")
    print(f"  Final val  perplexity : {val_ppl:.2f}")
    print("=" * 60)
    print("\n  Next: run  python server.py  for federated training.\n")
    print("  The server.py script will load baseline_history.json")
    print("  automatically and generate a comparison plot.\n")

    return model, history


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    train_centralized()
