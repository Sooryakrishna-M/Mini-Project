# =============================================================
# server.py — Flower Federated Learning Server & Orchestrator
# =============================================================
# This script is the entry-point for the entire FL experiment.
#
# WHAT HAPPENS HERE
# ─────────────────
#  1. GLOBAL DATA SETUP
#     • Load IMDB, build shared tokenizer, partition non-IID data.
#  2. FEDERATED TRAINING (5 rounds)
#     • Use fl.simulation.start_simulation() — all clients run in
#       the same process as separate threads (no network sockets
#       needed).  Perfect for Colab / laptops.
#     • Strategy: FedAvg — server computes weighted average of all
#       client weight updates each round.
#     • Custom strategy subclass logs per-round metrics.
#  3. GLOBAL MODEL SAVE
#     • Final aggregated weights → global_model.keras
#  4. PERSONALISATION PHASE
#     • Each client fine-tunes the Dense head on its private data.
#     • Saves client_i_model.keras for i in 0–9.
#  5. PLOTS & METRICS TABLE
#  6. DEMO INFERENCE — next-word prediction from seed text.
# =============================================================

import os
import json
import math
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, Scalar
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce

from model import build_model, VOCAB_SIZE, SEQ_LEN
from client import (
    get_client_fn,
    get_all_clients,
    get_tokenizer,
    MODELS_DIR,
    PERSONALIZATION_EPOCHS,
)
from utils import (
    compute_perplexity,
    metrics_table,
    plot_metrics,
    plot_comparison,
    plot_client_personalization,
    predict_next_words,
    load_imdb_raw,
    build_or_load_tokenizer,
    partition_non_iid,
    prepare_client_datasets,
    dataset_to_numpy,
    NUM_CLIENTS,
)

# ------------------------------------------------------------------
# FL Configuration
# ------------------------------------------------------------------
NUM_ROUNDS          = 5      # FL communication rounds
NUM_CLIENTS_FIT     = 5      # clients sampled per round (fraction_fit=1.0 of 5)
NUM_CLIENTS_EVAL    = 5      # clients sampled for evaluation
MIN_AVAILABLE       = 5      # minimum clients that must be available
PLOTS_DIR           = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==================================================================
# Custom FedAvg Strategy (adds logging)
# ==================================================================

class FedAvgWithLogging(fl.server.strategy.FedAvg):
    """
    FedAvg + per-round metric logging.

    FedAvg (McMahan et al. 2017):
      w_global ← Σ_k (n_k / n_total) * w_k
    where w_k are the weights returned by client k and n_k is the
    number of local samples used for training.

    This subclass adds:
      • Recording of aggregated metrics each round.
      • Printing a progress line per round.
    """

    def __init__(self, history: dict, **kwargs):
        super().__init__(**kwargs)
        self.history = history     # shared dict mutated in-place

    # ---- FIT AGGREGATION ----
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client fit results using FedAvg, then log metrics.
        """
        if failures:
            print(f"[Server] Round {server_round}: {len(failures)} client(s) failed.")

        # Delegate aggregation to parent FedAvg
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Collect per-client metrics sent back in fit()
        total_samples = 0
        weighted_loss = 0.0
        weighted_acc  = 0.0

        for _, fit_res in results:
            n = fit_res.num_examples
            total_samples += n
            weighted_loss += n * fit_res.metrics.get("loss",     0.0)
            weighted_acc  += n * fit_res.metrics.get("accuracy", 0.0)

        if total_samples > 0:
            avg_loss = weighted_loss / total_samples
            avg_acc  = weighted_acc  / total_samples
            avg_ppl  = compute_perplexity(avg_loss)

            self.history["round"].append(server_round)
            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(avg_acc)
            self.history["perplexity"].append(avg_ppl)

            print(
                f"\n[Server] ══ Round {server_round}/{NUM_ROUNDS} ══ "
                f"avg_loss={avg_loss:.4f}  avg_acc={avg_acc:.4f}  "
                f"ppl={avg_ppl:.2f}  (samples={total_samples})"
            )

        return aggregated_params, aggregated_metrics

    # ---- EVALUATE AGGREGATION ----
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluate results and log validation metrics.
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if results:
            total_samples = sum(r.num_examples for _, r in results)
            weighted_acc  = sum(
                r.num_examples * r.metrics.get("accuracy", 0.0)
                for _, r in results
            )
            avg_acc = weighted_acc / total_samples if total_samples > 0 else 0.0
            print(
                f"[Server] Round {server_round} EVAL — "
                f"val_loss={aggregated_loss:.4f}  val_acc={avg_acc:.4f}"
            )

        return aggregated_loss, aggregated_metrics


# ==================================================================
# Weight helpers
# ==================================================================

def get_global_model_weights(model: tf.keras.Model) -> Parameters:
    """Convert Keras model weights → Flower Parameters object."""
    ndarrays = model.get_weights()
    return fl.common.ndarrays_to_parameters(ndarrays)


def set_model_weights(model: tf.keras.Model,
                       parameters: Parameters) -> None:
    """Load Flower Parameters into a Keras model."""
    ndarrays = fl.common.parameters_to_ndarrays(parameters)
    model.set_weights(ndarrays)


# ==================================================================
# Evaluation callback (called by Flower each round)
# ==================================================================

def make_evaluate_fn(global_model: tf.keras.Model,
                     X_val: np.ndarray, y_val: np.ndarray):
    """
    Factory for the server-side evaluation function.

    Flower calls this each round to evaluate the aggregated global
    model on a held-out centralised validation set (test split).
    This is optional but gives us a clean picture of global progress.
    """
    def evaluate_fn(
        server_round: int,
        parameters,          # already a list of ndarrays here — NOT a Parameters object
        config: Dict,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Flower passes raw ndarrays to evaluate_fn, not a Parameters object
        global_model.set_weights(parameters)
        loss, accuracy = global_model.evaluate(
                X_val, y_val, batch_size=32, verbose=0
            )
        ppl = compute_perplexity(loss)
        print(
                f"[Server] Global eval (round {server_round}): "
                f"loss={loss:.4f}  acc={accuracy:.4f}  ppl={ppl:.2f}"
            )
        return loss, {"accuracy": accuracy, "perplexity": ppl}

        return evaluate_fn


# ==================================================================
# MAIN ORCHESTRATOR
# ==================================================================

def main():
    print("=" * 65)
    print("  Federated Next-Word Prediction — Flower + TensorFlow/Keras")
    print("=" * 65)

    # ── 1. Prepare shared data ─────────────────────────────────────
    print("\n[Phase 1] Loading & partitioning data …")
    tr_txt, tr_lbl, ts_txt, ts_lbl = load_imdb_raw()

    # Build shared tokenizer (saved to disk so clients can load it)
    tokenizer = build_or_load_tokenizer(tr_txt + ts_txt)

    # Prepare a small centralised validation set (test split, 2k reviews)
    from utils import texts_to_sequences
    print("[Server] Building validation set from test split …")
    X_val, y_val = texts_to_sequences(tokenizer, ts_txt[:300])
    print(f"[Server] Validation sequences: {len(X_val)}")

    # ── 2. Build global model ──────────────────────────────────────
    print("\n[Phase 2] Initialising global model …")
    global_model = build_model()
    global_model.summary()

    # Shared history dict (mutated by strategy each round)
    fed_history = {
        "round":      [],
        "loss":       [],
        "accuracy":   [],
        "perplexity": [],
    }

    # Initial parameters (random weights to broadcast on round 1)
    initial_parameters = get_global_model_weights(global_model)

    # ── 3. Define FedAvg strategy ──────────────────────────────────
    print("\n[Phase 3] Setting up FedAvg strategy …")
    strategy = FedAvgWithLogging(
        history=fed_history,
        fraction_fit=1.0,                  # sample ALL available clients
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS_FIT,
        min_evaluate_clients=NUM_CLIENTS_EVAL,
        min_available_clients=MIN_AVAILABLE,
        initial_parameters=initial_parameters,
        evaluate_fn=make_evaluate_fn(global_model, X_val, y_val),
        on_fit_config_fn=lambda r: {"server_round": r},
        on_evaluate_config_fn=lambda r: {"server_round": r},
    )

    # ── 4. Run federated simulation ────────────────────────────────
    print(f"\n[Phase 4] Starting FL simulation ({NUM_ROUNDS} rounds, "
          f"{NUM_CLIENTS_FIT} clients/round) …")
    print("  Note: All clients run in-process — no network required.\n")

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(),          # factory that returns TextPredClient
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={                   # CPU-only simulation
            "num_cpus": 1,
            "num_gpus": 0.0,
        },
    )

    # ── 5. Extract final global weights ────────────────────────────
    print("\n[Phase 5] Extracting final global model …")

    # Retrieve final round's aggregated parameters from Flower history
    # history.metrics_distributed_fit stores per-round aggregated metrics
    # The final parameters are in history.parameters (last entry)
    if hasattr(history, 'parameters') and history.parameters:
        final_params = history.parameters[-1]
        set_model_weights(global_model, final_params)
    else:
        # Fallback: re-run one round of evaluation to get consistent state
        print("[Server] Using current global_model weights (evaluate_fn kept them updated).")

    # Save global model
    global_model_path = os.path.join(MODELS_DIR, "global_model.keras")
    global_model.save(global_model_path)
    print(f"[Server] Global model saved → {global_model_path}")

    # Get final weights as numpy list for personalisation
    final_weights = global_model.get_weights()

    # ── 6. Print federated metrics table ───────────────────────────
    print("\n[Phase 6] Federated Learning Metrics:")
    metrics_table(fed_history)

    # ── 7. Plot FL metrics ─────────────────────────────────────────
    print("\n[Phase 7] Generating plots …")
    plot_metrics(fed_history, title_prefix="Federated", save_dir=PLOTS_DIR)

    # ── 8. Personalisation phase ───────────────────────────────────
    print(f"\n[Phase 8] Client personalisation ({PERSONALIZATION_EPOCHS} epochs each) …")
    all_clients = get_all_clients()
    client_results = {}

    for client in all_clients:
        result = client.personalize(
            global_weights=final_weights,
            epochs=PERSONALIZATION_EPOCHS,
        )
        client_results[client.client_id] = result

    # Print personalisation summary
    print("\n── Personalisation Summary ──")
    print(f"{'Client':>8} | {'Acc Before':>12} | {'Acc After':>12} | {'Δ Acc':>8}")
    print("-" * 48)
    for cid, r in sorted(client_results.items()):
        delta = r["after"] - r["before"]
        print(
            f"{cid:>8} | {r['before']:>12.4f} | {r['after']:>12.4f} | "
            f"{'+' if delta >= 0 else ''}{delta:>7.4f}"
        )

    # Plot per-client personalisation results
    plot_client_personalization(client_results, save_dir=PLOTS_DIR)

    # ── 9. Load centralized baseline and compare ───────────────────
    print("\n[Phase 9] Loading centralized baseline metrics for comparison …")
    baseline_path = "baseline_history.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            base_history = json.load(f)
        plot_comparison(fed_history, base_history, save_dir=PLOTS_DIR)
        print("  Comparison plot saved.")
    else:
        print(
            "  [Info] baseline_history.json not found.\n"
            "  Run  python baseline.py  first, then re-run server.py\n"
            "  for the comparison plot."
        )

    # ── 10. Demo inference ─────────────────────────────────────────
    print("\n[Phase 10] Demo — next-word prediction with global model")
    demo_seeds = [
        "the movie was really",
        "i loved this film because",
        "the acting was terrible and",
        "one of the best performances",
    ]
    print("\n─── Predictions (global model) ───")
    for seed in demo_seeds:
        prediction = predict_next_words(
            model=global_model,
            tokenizer=get_tokenizer(),
            seed_text=seed,
            num_words=5,
            temperature=0.8,
        )
        print(f"  Seed : '{seed}'")
        print(f"  →     '{prediction}'\n")

    # ── 11. Final summary ──────────────────────────────────────────
    if fed_history["accuracy"]:
        final_acc = fed_history["accuracy"][-1]
        final_ppl = fed_history["perplexity"][-1]
        avg_pers_gain = np.mean([
            r["after"] - r["before"] for r in client_results.values()
        ])
        print("=" * 65)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 65)
        print(f"  Global model accuracy  (round {NUM_ROUNDS}): {final_acc:.4f}")
        print(f"  Global model perplexity(round {NUM_ROUNDS}): {final_ppl:.2f}")
        print(f"  Avg personalisation gain              : +{avg_pers_gain:.4f}")
        print(f"  Saved models in: {MODELS_DIR}/")
        print(f"  Saved plots  in: {PLOTS_DIR}/")
        print("=" * 65)

    print("\n✓  Federated training complete!\n")


if __name__ == "__main__":
    # Suppress TF logging noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
