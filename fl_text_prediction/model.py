# =============================================================
# model.py — Shared LSTM Model Definition
# =============================================================
# This file defines the LSTM model used by ALL clients and the
# server. Keeping it in one place guarantees every participant
# uses identical architecture, which is critical for Federated
# Averaging (weights must have the same shape everywhere).
#
# Architecture:
#   Input  : (batch, 50) integer token IDs
#   Embed  : 10,000-vocab → 128-dim dense vectors
#   LSTM   : 128 hidden units, many-to-one (return_sequences=False)
#   Output : Dense(10,000) with softmax → next-word probability dist
# =============================================================

import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------------
# Hyper-parameters (shared across all files via import)
# ------------------------------------------------------------------
VOCAB_SIZE   = 10_000   # top-N words kept from IMDB vocabulary
SEQ_LEN      = 50       # fixed input sequence length (padded/truncated)
EMBED_DIM    = 128      # embedding dimensionality
LSTM_UNITS   = 128      # number of LSTM hidden units
BATCH_SIZE   = 32       # mini-batch size used during training
LEARNING_RATE = 1e-3    # Adam optimizer learning rate


def build_model(vocab_size: int = VOCAB_SIZE,
                seq_len: int = SEQ_LEN,
                embed_dim: int = EMBED_DIM,
                lstm_units: int = LSTM_UNITS,
                learning_rate: float = LEARNING_RATE) -> keras.Model:
    """
    Build and compile the LSTM next-word prediction model.

    The model predicts the probability distribution over the entire
    vocabulary for the token that follows the input sequence.

    Parameters
    ----------
    vocab_size     : number of unique tokens (including OOV token at index 1)
    seq_len        : expected input sequence length
    embed_dim      : width of the embedding layer
    lstm_units     : number of recurrent units in the LSTM layer
    learning_rate  : Adam learning rate

    Returns
    -------
    compiled keras.Model ready for training
    """
    # --- Input layer ---
    # Each sample is a sequence of SEQ_LEN integer token IDs.
    inputs = keras.Input(shape=(seq_len,), name="token_ids")

    # --- Embedding layer ---
    # Maps each integer token ID → a trainable dense vector.
    # mask_zero=True tells downstream layers to ignore padding (0 tokens).
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,           # padding index = 0 is masked
        name="embedding"
    )(inputs)

    # --- LSTM layer ---
    # Processes the sequence left-to-right and returns only the
    # final hidden state (return_sequences=False → shape: (batch, lstm_units)).
    # dropout / recurrent_dropout add regularisation.
    x = keras.layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.1,
        name="lstm"
    )(x)

    # --- Output Dense layer ---
    # Projects LSTM output → vocabulary-sized logits, then softmax
    # converts them to a proper probability distribution.
    outputs = keras.layers.Dense(
        units=vocab_size,
        activation="softmax",
        name="next_word_probs"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_NextWord")

    # --- Compile ---
    # sparse_categorical_crossentropy: targets are integer class indices
    # (not one-hot), which saves memory for large vocabularies.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_personalized_model(base_model: keras.Model,
                              learning_rate: float = 5e-4) -> keras.Model:
    """
    Prepare a model for client-side personalisation.

    Strategy: freeze all layers EXCEPT the final Dense head, then
    re-compile with a lower learning rate.  This lets each client
    adapt output probabilities to its own vocabulary distribution
    without forgetting the shared representations learned globally.

    Parameters
    ----------
    base_model    : the global model whose weights we start from
    learning_rate : (smaller) LR for fine-tuning

    Returns
    -------
    keras.Model with head unfrozen, body frozen
    """
    # Freeze embedding + LSTM
    for layer in base_model.layers[:-1]:   # everything except last layer
        layer.trainable = False

    # The Dense head remains trainable
    base_model.layers[-1].trainable = True

    base_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return base_model


if __name__ == "__main__":
    # Quick sanity check: build, summarise, and do a forward pass.
    import numpy as np

    print("Building model …")
    model = build_model()
    model.summary()

    # Random input: batch of 4 sequences, each of length SEQ_LEN
    dummy_input  = np.random.randint(0, VOCAB_SIZE, size=(4, SEQ_LEN))
    dummy_output = model.predict(dummy_input, verbose=0)

    print(f"\nInput  shape : {dummy_input.shape}")
    print(f"Output shape : {dummy_output.shape}")   # expected: (4, 10000)
    print(f"Prob sums    : {dummy_output.sum(axis=-1)}")  # should be ≈ [1,1,1,1]
    print("\n✓  model.py  sanity-check passed.")
