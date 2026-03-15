# =============================================================
# client.py — Flower Federated Learning Client
# =============================================================
# Each simulated device (client) is an instance of TextPredClient.
# Flower calls fit() and evaluate() each round via the strategy.
#
# PRIVACY GUARANTEE
# ─────────────────
# The server NEVER receives raw text, token sequences, or labels.
# Only model weight *updates* (numpy arrays) are sent upward.
# The server aggregates these updates with FedAvg (weighted mean)
# and broadcasts the new global weights back.  This is the core
# of Federated Learning's privacy model.
#
# CLIENT LIFECYCLE PER ROUND
# ───────────────────────────
#   1. Server calls get_parameters() → client sends current weights.
#   2. Server sends global weights → client calls set_parameters().
#   3. Server calls fit()           → client trains 2 local epochs.
#   4. Server calls evaluate()      → client returns loss + accuracy.
# =============================================================

import os
import math
import pickle
import numpy as np
import tensorflow as tf
import flwr as fl
from typing import Dict, List, Tuple

from model import build_model, VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, LEARNING_RATE
from utils  import (
    load_imdb_raw,
    build_or_load_tokenizer,
    partition_non_iid,
    prepare_client_datasets,
    dataset_to_numpy,
    compute_perplexity,
    NUM_CLIENTS,
    TOKENIZER_PATH,
)

# ------------------------------------------------------------------
# FL hyper-parameters
# ------------------------------------------------------------------
LOCAL_EPOCHS       = 2     # how many epochs each client trains per FL round
PERSONALIZATION_EPOCHS = 5 # fine-tuning epochs after global training ends
MODELS_DIR         = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ==================================================================
# Flower NumPyClient
# ==================================================================

class TextPredClient(fl.client.NumPyClient):
    """
    Flower client for next-word prediction.

    Each instance holds:
      • A local Keras LSTM model.
      • A local tf.data.Dataset (its private data partition).
      • Its unique client_id (0–9).

    Parameters
    ----------
    client_id : int
        Unique identifier for this simulated device.
    dataset   : tf.data.Dataset
        Batched (X, y) sequences belonging to this client.
    """

    def __init__(self, client_id: int, dataset: tf.data.Dataset):
        self.client_id = client_id
        self.dataset   = dataset
        self.model     = build_model()   # fresh model; weights set by server each round

        # Pull numpy arrays once (used in fit/evaluate)
        self.X, self.y = dataset_to_numpy(dataset)
        print(f"[Client {client_id}] Initialised | sequences: {len(self.X)}")

    # ------------------------------------------------------------------
    # Flower interface — weight transfer
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return the current model weights as a list of numpy arrays.
        Called by Flower to collect weights from this client.
        """
        return self.model.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load the aggregated global weights sent by the server.
        Called by Flower before each round's fit() and evaluate().
        """
        self.model.set_weights(parameters)

    # ------------------------------------------------------------------
    # Flower interface — local training
    # ------------------------------------------------------------------

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receive global model weights, train locally, return updated weights.

        Parameters
        ----------
        parameters : global weights from server
        config     : dict with optional round info

        Returns
        -------
        (updated_weights, num_samples, metrics_dict)
          • num_samples — used by FedAvg for weighted aggregation
          • metrics_dict — optional per-client metrics logged by server
        """
        # 1. Load global weights
        self.set_parameters(parameters)

        # 2. Local training (LOCAL_EPOCHS epochs, private data)
        history = self.model.fit(
            self.X, self.y,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,    # 10% local validation
            verbose=0,               # suppress per-epoch output in sim
        )

        # Extract final epoch metrics
        loss     = float(history.history["loss"][-1])
        accuracy = float(history.history["accuracy"][-1])
        ppl      = compute_perplexity(loss)

        fl_round = config.get("server_round", "?")
        print(
            f"  [Client {self.client_id}] Round {fl_round} | "
            f"loss={loss:.4f} | acc={accuracy:.4f} | ppl={ppl:.2f}"
        )

        # 3. Return updated weights + metadata
        return (
            self.model.get_weights(),
            len(self.X),                        # weighted aggregation
            {
                "loss":       loss,
                "accuracy":   accuracy,
                "perplexity": ppl,
            },
        )

    # ------------------------------------------------------------------
    # Flower interface — local evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the global model on this client's local data.

        This is called by the server after aggregation to measure
        how well the new global model generalises to each client.

        Returns
        -------
        (loss, num_samples, metrics_dict)
        """
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(
            self.X, self.y,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        ppl = compute_perplexity(loss)
        return float(loss), len(self.X), {"accuracy": accuracy, "perplexity": ppl}

    # ------------------------------------------------------------------
    # Post-FL personalisation
    # ------------------------------------------------------------------

    def personalize(self, global_weights: List[np.ndarray],
                    epochs: int = PERSONALIZATION_EPOCHS) -> Dict:
        """
        Fine-tune the final global model on this client's private data.

        Strategy: freeze all layers except the Dense output head.
        This preserves shared representations while adapting the
        output distribution to the client's unique vocabulary usage.

        Parameters
        ----------
        global_weights : weights from the final global model
        epochs         : fine-tuning epochs

        Returns
        -------
        dict with 'before' and 'after' accuracy, loss, perplexity
        """
        from model import build_personalized_model

        # Evaluate BEFORE personalisation
        self.set_parameters(global_weights)
        loss_before, acc_before = self.model.evaluate(
            self.X, self.y, batch_size=BATCH_SIZE, verbose=0
        )

        # Freeze body, unfreeze head
        personalized = build_personalized_model(self.model)

        # Fine-tune on private data
        personalized.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        # Evaluate AFTER personalisation
        loss_after, acc_after = personalized.evaluate(
            self.X, self.y, batch_size=BATCH_SIZE, verbose=0
        )

        # Save the personalised model
        save_path = os.path.join(MODELS_DIR, f"client_{self.client_id}_model.keras")
        personalized.save(save_path)
        print(
            f"  [Client {self.client_id}] Personalisation | "
            f"acc {acc_before:.4f} → {acc_after:.4f} | "
            f"Saved: {save_path}"
        )

        return {
            "before": acc_before,
            "after":  acc_after,
            "loss_before": loss_before,
            "loss_after":  loss_after,
            "ppl_before":  compute_perplexity(loss_before),
            "ppl_after":   compute_perplexity(loss_after),
        }


# ==================================================================
# Client factory — used by server.py simulation loop
# ==================================================================

# These are module-level caches so we only load data once per process
_datasets: Dict[int, tf.data.Dataset] = {}
_tokenizer = None


def _ensure_data_loaded():
    """Load IMDB data and build client datasets (called once per process)."""
    global _datasets, _tokenizer

    if _datasets:
        return  # already loaded

    tr_txt, tr_lbl, ts_txt, ts_lbl = load_imdb_raw()
    _tokenizer = build_or_load_tokenizer(tr_txt + ts_txt)
    client_raw = partition_non_iid(tr_txt, tr_lbl, num_clients=NUM_CLIENTS)
    _datasets  = prepare_client_datasets(client_raw, _tokenizer)


def get_client_fn():
    """
    Returns a closure that Flower calls to instantiate clients.

    Flower passes a string `cid` (client id).  We convert it to int
    and return the pre-built TextPredClient for that partition.
    """
    _ensure_data_loaded()

    def client_fn(cid: str) -> fl.client.NumPyClient:
        cid_int = int(cid)
        if cid_int not in _datasets:
            raise ValueError(f"No dataset for client {cid_int}")
        return TextPredClient(client_id=cid_int, dataset=_datasets[cid_int])

    return client_fn


def get_all_clients() -> List[TextPredClient]:
    """Return all 10 client objects (used for personalisation phase)."""
    _ensure_data_loaded()
    return [TextPredClient(cid, _datasets[cid]) for cid in sorted(_datasets)]


def get_tokenizer():
    """Return the shared tokenizer (loaded alongside datasets)."""
    _ensure_data_loaded()
    return _tokenizer


# ==================================================================
# Standalone entry-point (single-client test)
# ==================================================================
if __name__ == "__main__":
    print("=== client.py standalone test ===")
    _ensure_data_loaded()

    # Test a single client for 1 round
    client = TextPredClient(client_id=0, dataset=_datasets[0])
    weights = client.get_parameters({})
    print(f"Model has {len(weights)} weight tensors.")
    print(f"Weight shapes: {[w.shape for w in weights]}")

    # Simulate one fit round
    updated_w, n_samples, metrics = client.fit(weights, {"server_round": 1})
    print(f"\nFit result — samples={n_samples}, metrics={metrics}")

    # Simulate evaluate
    loss, n_eval, eval_metrics = client.evaluate(updated_w, {})
    print(f"Eval result — loss={loss:.4f}, metrics={eval_metrics}")
    print("\n✓  client.py standalone test passed.")
