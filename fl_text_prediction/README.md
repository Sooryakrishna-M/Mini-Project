# 🧠 Federated Next-Word Text Prediction
### Personalized Keyboard App Simulation with Flower + TensorFlow/Keras

---

## 📖 Project Overview

This project demonstrates **Federated Learning (FL)** for personalized next-word prediction — the same technology powering predictive text on smartphones. The key insight: your phone's keyboard can learn *your* typing style without ever sending your private messages to a server.

**How it works:**
1. 10 simulated "mobile devices" (clients) each hold private text data (IMDB reviews)
2. Each client trains a local LSTM model and shares only **weight updates** (not data) with the server
3. The server aggregates updates via **FedAvg** → new global model
4. After 5 rounds, each client **personalizes** the global model to its own vocabulary
5. No raw text ever leaves any client — **privacy by design**

---

## 🗂️ Project Structure

```
fl_text_prediction/
├── README.md          ← You are here
├── requirements.txt   ← Python dependencies
├── model.py           ← Shared LSTM architecture (Embedding→LSTM→Dense)
├── utils.py           ← Data loading, tokenizer, non-IID split, plots
├── client.py          ← Flower NumPyClient (fit/evaluate/personalize)
├── server.py          ← FL orchestrator: simulation, strategy, save models
└── baseline.py        ← Centralized training for comparison
```

**Generated outputs (after running):**
```
saved_models/
├── global_model.keras           ← Final federated global model
├── centralized_model.keras      ← Baseline centralized model
├── client_0_model.keras         ← Personalized model for client 0
├── client_1_model.keras         ← ... (clients 0–9)
└── ...

plots/
├── federated_loss.png           ← FL loss per round
├── federated_accuracy.png       ← FL accuracy per round
├── centralized_loss.png         ← Baseline loss per epoch
├── centralized_accuracy.png     ← Baseline accuracy per epoch
├── comparison_fed_vs_centralized.png  ← Side-by-side comparison
└── client_personalization.png   ← Per-client accuracy before/after

tokenizer.pkl                    ← Shared vocabulary (built once)
baseline_history.json            ← Centralized metrics (for comparison)
```

---

## ⚙️ Setup Instructions

### Option A: Standard Python Environment

```bash
# 1. Clone / download the project
cd fl_text_prediction

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project (see below)
```

### Option B: Google Colab

```python
# Cell 1: Install dependencies
!pip install flwr==1.7.0 tensorflow==2.15.0 tensorflow-datasets==4.9.4 numpy==1.26.4 matplotlib==3.8.4 tqdm==4.66.2

# Cell 2: Upload files or clone repo
# (upload all .py files to Colab session)

# Cell 3: Run baseline
%run baseline.py

# Cell 4: Run federated training
%run server.py
```

### Option C: Conda

```bash
conda create -n fl_text python=3.10
conda activate fl_text
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Centralized Baseline (optional but recommended)
```bash
python baseline.py
```
Trains a standard LSTM on all data. Takes ~10–15 minutes.
Saves `saved_models/centralized_model.keras` and `baseline_history.json`.

### Step 2: Federated Training + Personalization
```bash
python server.py
```
Runs 5 FL rounds with 10 simulated clients, then personalizes each client.
Takes ~20–40 minutes depending on hardware.

### Step 3: Standalone Tests (optional)
```bash
python model.py      # Sanity check: build model, forward pass
python utils.py      # Sanity check: load data, tokenizer, partition
python client.py     # Sanity check: single client, 1 round
```

---

## 🏗️ Architecture

### LSTM Model
```
Input: (batch, 50) — integer token IDs
  ↓
Embedding(10000 → 128, mask_zero=True)
  ↓
LSTM(128, dropout=0.2, recurrent_dropout=0.1)
  ↓
Dense(10000, softmax)
  ↓
Output: (batch, 10000) — next-word probability distribution
```

**Parameters:** ~2.8M total
- Embedding layer: 10,000 × 128 = 1,280,000
- LSTM layer: ~132,096
- Dense head: 128 × 10,000 + 10,000 = 1,290,000

### Non-IID Data Partitioning
```
IMDB 25k train reviews
├── Positive reviews (12,500) → Clients 0–4  (2,500 each)
└── Negative reviews (12,500) → Clients 5–9  (2,500 each)
```
Each client sees a **different slice** of its sentiment group → unique vocabulary distribution → true non-IID setup.

### Federated Averaging (FedAvg)
```
Round r:
  For each client k:
    w_k ← w_global                    # receive global weights
    w_k ← local_sgd(w_k, D_k, epochs=2)  # 2 epochs on private data
    send w_k to server                 # only weights, not D_k!

  w_global ← Σ_k (n_k / Σ n_k) * w_k  # weighted average
```

### Personalization Strategy
After global FL completes:
1. Each client starts from global weights
2. **Freeze** Embedding + LSTM layers (shared knowledge)
3. **Fine-tune** only the Dense head (5 epochs, local data)
4. Save as `client_i_model.keras`

This approach (Transfer Learning / Head Fine-tuning) is efficient and avoids catastrophic forgetting.

---

## 📊 Expected Results

| Metric | Federated (5 rounds) | Centralized (5 epochs) |
|--------|---------------------|----------------------|
| Train loss | ~4.5–5.5 | ~3.5–4.5 |
| Train accuracy | ~30–40% | ~40–55% |
| Perplexity | ~90–250 | ~33–90 |
| Val accuracy | ~28–38% | ~38–50% |

**Personalization improvement:** +2–8% accuracy per client (varies by data size)

> ⚠️ Note: IMDB is a sentiment dataset, not a natural language corpus optimized for next-word prediction. Perplexity will be higher than on e.g. Wikipedia. This is expected and demonstrates the non-IID challenge realistically.

**The "federation gap"** (centralized minus federated accuracy) is typically 5–15%, which represents the cost of privacy. In production systems with more rounds and clients, this gap narrows significantly.

---

## 🔐 Privacy Guarantees

| What leaves the client? | What stays on the client? |
|------------------------|--------------------------|
| ✅ Model weight updates (numpy arrays) | 🔒 Raw text / reviews |
| ✅ Aggregate metrics (loss, acc) | 🔒 Token sequences |
| ❌ Never: raw data | 🔒 Individual words typed |

**Additional protections possible** (not implemented here, extensions welcome):
- **Differential Privacy**: Add calibrated Gaussian noise to weight updates (use `flwr` + `tensorflow_privacy`)
- **Secure Aggregation**: Cryptographic protocol so server sees only the sum, not individual updates
- **Homomorphic Encryption**: Aggregate encrypted weights

---

## 🧩 Key Concepts Explained

### Federated Learning vs Centralized
```
Centralized:          Federated:
  Client A ─── data ──► Server    Client A ─── Δweights ──► Server
  Client B ─── data ──► Server    Client B ─── Δweights ──► Server
  Server trains once              Server aggregates → broadcasts back
  Privacy: ❌ data exposed       Privacy: ✅ data stays local
```

### Why Non-IID Matters
In real FL, data is rarely uniform across clients:
- User A writes formal business emails
- User B sends casual texts with slang
- User C writes in multiple languages

Our partition simulates this: positive-review clients develop different embedding representations than negative-review clients, making aggregation harder but more realistic.

### Perplexity
```
Perplexity = exp(cross-entropy loss)

Interpretation:
  PPL = 10,000  → model is as good as random guessing (uniform over 10k words)
  PPL = 100     → model has narrowed candidates to ~100 plausible next words
  PPL = 10      → excellent model, very confident predictions
```

---

## 🛠️ Extending the Project

### Add Differential Privacy
```python
# In client.py fit():
from tensorflow_privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=0.1,
    num_microbatches=1,
    learning_rate=1e-3
)
```

### Use Your Own Data
Replace `load_imdb_raw()` in `utils.py` with your own corpus:
```python
def load_custom_data():
    # Load from files, databases, etc.
    return texts, labels
```

### More Clients / Rounds
In `server.py`:
```python
NUM_ROUNDS      = 10   # more rounds
NUM_CLIENTS_FIT = 20   # more clients per round
```
In `utils.py`:
```python
NUM_CLIENTS = 20       # total simulated devices
```

### Async Federated Learning
```python
strategy = fl.server.strategy.FedAsync(...)  # instead of FedAvg
```

### Flower with Real Network
```bash
# Terminal 1: Start server
flower-server --server-address 0.0.0.0:8080

# Terminal 2+: Start real clients
python -c "
import flwr as fl
from client import TextPredClient, _datasets, _tokenizer
fl.client.start_numpy_client(
    server_address='localhost:8080',
    client=TextPredClient(0, _datasets[0])
)
"
```

---

## 📚 References

1. McMahan et al. (2017) — "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg paper): https://arxiv.org/abs/1602.05629
2. Flower Framework: https://flower.dev
3. TensorFlow Federated: https://www.tensorflow.org/federated
4. "Towards Federated Learning at Scale" (Google): https://arxiv.org/abs/1902.01046
5. Personalized Federated Learning survey: https://arxiv.org/abs/2103.00710

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: flwr` | Run `pip install flwr==1.7.0` |
| `TF version mismatch` | Run `pip install tensorflow==2.15.0` |
| OOM / memory error | Reduce `BATCH_SIZE` in `model.py` to 16 |
| Slow training | Set `NUM_CLIENTS_FIT=3`, `NUM_ROUNDS=3` in `server.py` |
| TFDS download fails | Check internet; IMDB is ~80MB |
| `tokenizer.pkl` missing | Delete it and re-run; it auto-regenerates |
| Colab disconnects | Save checkpoints; use `model.save()` in callbacks |

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.

---

*Built with ❤️ using [Flower](https://flower.dev) + [TensorFlow/Keras](https://tensorflow.org)*
