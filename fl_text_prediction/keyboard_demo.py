# keyboard_demo.py
# ─────────────────────────────────────────────────────────────────
# Flask backend: serves predictions from global + personalized models
# Run: python keyboard_demo.py
# Then open: http://localhost:5000
# ─────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import predict_next_words, build_or_load_tokenizer, load_imdb_raw

app = Flask(__name__)

# ── Load models & tokenizer once at startup ──────────────────────
print("Loading tokenizer...")
try:
    tokenizer = build_or_load_tokenizer()
except:
    print("Tokenizer not found, loading IMDB to build it...")
    tr, _, ts, _ = load_imdb_raw()
    tokenizer = build_or_load_tokenizer(tr + ts)

print("Loading global model...")
global_model = tf.keras.models.load_model("saved_models/global_model.keras")

print("Loading personalized models...")
client_models = {}
for i in range(10):
    path = f"saved_models/client_{i}_model.keras"
    if os.path.exists(path):
        client_models[i] = tf.keras.models.load_model(path)
        print(f"  Loaded client_{i}_model.keras")

print(f"\n✓ Ready! Open http://localhost:5000\n")

# ── HTML (single-file app) ────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FL Keyboard Demo</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0f0f13;
    --surface: #1a1a24;
    --surface2: #22222f;
    --border: #2e2e3f;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --accent3: #43e97b;
    --text: #e8e8f0;
    --text2: #8888aa;
    --key-bg: #252535;
    --key-hover: #2e2e45;
    --key-active: #6c63ff;
    --shadow: 0 8px 32px rgba(0,0,0,0.4);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px 40px;
    background-image: radial-gradient(ellipse at 20% 10%, rgba(108,99,255,0.08) 0%, transparent 50%),
                      radial-gradient(ellipse at 80% 80%, rgba(255,101,132,0.06) 0%, transparent 50%);
  }

  /* ── Header ── */
  .header {
    text-align: center;
    margin-bottom: 28px;
    animation: fadeDown 0.6s ease;
  }
  .header h1 {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.3px;
    color: var(--text);
  }
  .header h1 span { color: var(--accent); }
  .header p {
    font-size: 12px;
    color: var(--text2);
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
  }

  /* ── Phone shell ── */
  .phone {
    width: 360px;
    background: var(--surface);
    border-radius: 36px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow), 0 0 0 1px rgba(255,255,255,0.03);
    overflow: hidden;
    animation: fadeUp 0.7s ease;
  }

  /* ── Status bar ── */
  .statusbar {
    background: var(--surface2);
    padding: 10px 20px 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: var(--text2);
    font-family: 'DM Mono', monospace;
  }
  .statusbar .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent3);
    display: inline-block;
    margin-right: 5px;
    animation: pulse 2s infinite;
  }

  /* ── App bar ── */
  .appbar {
    background: var(--surface2);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid var(--border);
  }
  .app-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }
  .app-name { font-size: 13px; font-weight: 600; color: var(--text); }
  .app-sub  { font-size: 10px; color: var(--text2); font-family: 'DM Mono', monospace; }

  /* ── Message area ── */
  .messages {
    min-height: 160px;
    padding: 14px 14px 6px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
    max-height: 200px;
  }
  .bubble {
    max-width: 82%;
    padding: 9px 13px;
    border-radius: 18px;
    font-size: 13.5px;
    line-height: 1.4;
    animation: popIn 0.25s ease;
  }
  .bubble.sent {
    align-self: flex-end;
    background: var(--accent);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .bubble.recv {
    align-self: flex-start;
    background: var(--surface2);
    color: var(--text);
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
  }
  .bubble .label {
    font-size: 9px;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
    font-family: 'DM Mono', monospace;
  }

  /* ── Typing area ── */
  .typing-area {
    padding: 8px 12px 4px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .input-box {
    flex: 1;
    background: var(--key-bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 9px 14px;
    font-size: 14px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    outline: none;
    transition: border-color 0.2s;
    min-height: 38px;
  }
  .input-box:focus { border-color: var(--accent); }
  .input-box::placeholder { color: var(--text2); }

  .send-btn {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    transition: transform 0.15s, background 0.2s;
    flex-shrink: 0;
  }
  .send-btn:hover { transform: scale(1.08); background: #7b74ff; }
  .send-btn:active { transform: scale(0.95); }

  /* ── Suggestion bar ── */
  .suggestion-bar {
    background: var(--surface2);
    padding: 8px 10px;
    display: flex;
    gap: 6px;
    min-height: 46px;
    align-items: center;
    border-top: 1px solid var(--border);
    overflow-x: auto;
  }
  .suggestion-bar::-webkit-scrollbar { display: none; }

  .suggestion {
    background: var(--key-bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 14px;
    border-radius: 10px;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
    font-family: 'DM Sans', sans-serif;
    position: relative;
    overflow: hidden;
  }
  .suggestion:hover {
    background: var(--key-hover);
    border-color: var(--accent);
    transform: translateY(-1px);
  }
  .suggestion:active { transform: scale(0.96); }
  .suggestion.global  { border-left: 3px solid var(--accent); }
  .suggestion.personal { border-left: 3px solid var(--accent3); }
  .suggestion.loading {
    opacity: 0.5;
    pointer-events: none;
  }

  .suggestion-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--text2);
    font-family: 'DM Mono', monospace;
    margin-right: 2px;
    flex-shrink: 0;
  }

  /* ── Keyboard rows ── */
  .keyboard {
    background: var(--bg);
    padding: 8px 6px 10px;
  }
  .kb-row {
    display: flex;
    justify-content: center;
    gap: 5px;
    margin-bottom: 5px;
  }
  .key {
    background: var(--key-bg);
    border: none;
    border-radius: 7px;
    color: var(--text);
    font-size: 14px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    height: 40px;
    min-width: 30px;
    flex: 1;
    max-width: 34px;
    cursor: pointer;
    transition: all 0.1s;
    box-shadow: 0 2px 0 rgba(0,0,0,0.4);
    display: flex; align-items: center; justify-content: center;
    user-select: none;
  }
  .key:hover { background: var(--key-hover); }
  .key:active {
    background: var(--accent);
    transform: translateY(1px);
    box-shadow: 0 1px 0 rgba(0,0,0,0.4);
    color: white;
  }
  .key.wide   { max-width: 52px; font-size: 11px; }
  .key.space  { max-width: 160px; font-size: 11px; color: var(--text2); }
  .key.action {
    background: var(--accent);
    color: white;
    max-width: 52px;
    font-size: 11px;
  }
  .key.backspace { max-width: 46px; font-size: 16px; }
  .key.enter { max-width: 52px; font-size: 11px; background: var(--accent2); color: white; }

  /* ── Client selector ── */
  .client-selector {
    width: 360px;
    margin-top: 16px;
    background: var(--surface);
    border-radius: 16px;
    border: 1px solid var(--border);
    padding: 14px 16px;
    animation: fadeUp 0.8s ease;
  }
  .selector-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text2);
    font-family: 'DM Mono', monospace;
    margin-bottom: 10px;
  }
  .client-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
  }
  .client-btn {
    background: var(--key-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 4px;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
    color: var(--text);
  }
  .client-btn:hover { border-color: var(--accent); background: var(--key-hover); }
  .client-btn.active {
    border-color: var(--accent3);
    background: rgba(67,233,123,0.1);
  }
  .client-btn .cnum {
    font-size: 16px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
  }
  .client-btn .ctag {
    font-size: 9px;
    color: var(--text2);
    margin-top: 2px;
  }
  .client-btn.active .ctag { color: var(--accent3); }

  /* ── Legend ── */
  .legend {
    width: 360px;
    margin-top: 10px;
    display: flex;
    gap: 14px;
    justify-content: center;
    animation: fadeUp 0.9s ease;
  }
  .legend-item {
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; color: var(--text2);
    font-family: 'DM Mono', monospace;
  }
  .legend-dot {
    width: 10px; height: 10px;
    border-radius: 2px;
  }

  /* ── Animations ── */
  @keyframes fadeDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes popIn {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }
  @keyframes shimmer {
    0%   { background-position: -200px 0; }
    100% { background-position: 200px 0; }
  }
  .shimmer {
    background: linear-gradient(90deg, var(--key-bg) 25%, var(--key-hover) 50%, var(--key-bg) 75%);
    background-size: 400px 100%;
    animation: shimmer 1.2s infinite;
  }
</style>
</head>
<body>

<div class="header">
  <h1>⌨️ FL <span>Keyboard</span> Demo</h1>
  <p>Federated Learning · Next-Word Prediction · ICCS College</p>
</div>

<!-- Phone -->
<div class="phone">

  <!-- Status bar -->
  <div class="statusbar">
    <span><span class="dot"></span>FL Active</span>
    <span>9:41 AM</span>
    <span>●●● 100%</span>
  </div>

  <!-- App bar -->
  <div class="appbar">
    <div class="app-icon">💬</div>
    <div>
      <div class="app-name">Messages</div>
      <div class="app-sub" id="client-label">Global Model</div>
    </div>
  </div>

  <!-- Message bubbles -->
  <div class="messages" id="messages">
    <div class="bubble recv">
      <div class="label">System</div>
      Start typing below to see FL predictions!
    </div>
  </div>

  <!-- Typing area -->
  <div class="typing-area">
    <input
      class="input-box"
      id="textInput"
      type="text"
      placeholder="Type a message..."
      autocomplete="off"
    />
    <button class="send-btn" onclick="sendMessage()" title="Send">➤</button>
  </div>

  <!-- Suggestion bar -->
  <div class="suggestion-bar" id="suggestionBar">
    <span class="suggestion-label">Suggestions</span>
    <div class="suggestion global shimmer" style="width:70px">&nbsp;</div>
    <div class="suggestion global shimmer" style="width:55px">&nbsp;</div>
    <div class="suggestion personal shimmer" style="width:65px">&nbsp;</div>
  </div>

  <!-- Keyboard -->
  <div class="keyboard">
    <div class="kb-row" id="row1"></div>
    <div class="kb-row" id="row2"></div>
    <div class="kb-row" id="row3"></div>
    <div class="kb-row" id="row4"></div>
  </div>

</div>

<!-- Client selector -->
<div class="client-selector">
  <div class="selector-title">Select Client / Device</div>
  <div class="client-grid" id="clientGrid"></div>
</div>

<!-- Legend -->
<div class="legend">
  <div class="legend-item">
    <div class="legend-dot" style="background:var(--accent)"></div>
    Global Model
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:var(--accent3)"></div>
    Personalized
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────
let selectedClient = null;   // null = global only
let debounceTimer  = null;

// ── Keyboard layout ───────────────────────────────────────────
const rows = [
  ['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l'],
  ['⇧','z','x','c','v','b','n','m','⌫'],
  ['123','⎵ space','↵']
];
const rowIds = ['row1','row2','row3','row4'];

function buildKeyboard() {
  rows.forEach((row, ri) => {
    const el = document.getElementById(rowIds[ri]);
    row.forEach(k => {
      const btn = document.createElement('button');
      btn.className = 'key';
      if (k === '⌫')     { btn.className += ' backspace'; btn.textContent = '⌫'; }
      else if (k === '⎵ space') { btn.className += ' space'; btn.textContent = 'space'; }
      else if (k === '↵') { btn.className += ' enter';  btn.textContent = 'return'; }
      else if (k === '⇧') { btn.className += ' wide';   btn.textContent = '⇧'; }
      else if (k === '123'){ btn.className += ' wide action'; btn.textContent = '123'; }
      else                { btn.textContent = k; }

      btn.addEventListener('click', () => handleKey(k));
      el.appendChild(btn);
    });
  });
}

function handleKey(k) {
  const inp = document.getElementById('textInput');
  if (k === '⌫') {
    inp.value = inp.value.slice(0, -1);
  } else if (k === '⎵ space') {
    inp.value += ' ';
  } else if (k === '↵') {
    sendMessage(); return;
  } else if (k === '⇧' || k === '123') {
    return;
  } else {
    inp.value += k;
  }
  inp.focus();
  triggerPrediction();
}

// ── Client grid ───────────────────────────────────────────────
function buildClientGrid() {
  const grid = document.getElementById('clientGrid');
  const sentiments = ['😊','😊','😊','😊','😊','😠','😠','😠','😠','😠'];
  const labels     = ['Pos','Pos','Pos','Pos','Pos','Neg','Neg','Neg','Neg','Neg'];

  // Global option
  const gBtn = document.createElement('div');
  gBtn.className = 'client-btn active';
  gBtn.id = 'client-global';
  gBtn.innerHTML = `<div class="cnum">🌐</div><div class="ctag">Global</div>`;
  gBtn.onclick = () => selectClient(null);
  grid.appendChild(gBtn);

  for (let i = 0; i < 10; i++) {
    const btn = document.createElement('div');
    btn.className = 'client-btn';
    btn.id = `client-${i}`;
    btn.innerHTML = `<div class="cnum">${sentiments[i]}${i}</div><div class="ctag">${labels[i]}</div>`;
    btn.onclick = () => selectClient(i);
    grid.appendChild(btn);
  }
}

function selectClient(id) {
  // Remove active from all
  document.querySelectorAll('.client-btn').forEach(b => b.classList.remove('active'));
  const key = id === null ? 'client-global' : `client-${id}`;
  document.getElementById(key).classList.add('active');

  selectedClient = id;
  const label = id === null ? 'Global Model' : `Client ${id} · Personalized`;
  document.getElementById('client-label').textContent = label;

  triggerPrediction();
}

// ── Prediction ────────────────────────────────────────────────
function triggerPrediction() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchPredictions, 280);
}

function showLoadingSuggestions() {
  const bar = document.getElementById('suggestionBar');
  bar.innerHTML = `<span class="suggestion-label">Loading</span>
    <div class="suggestion shimmer" style="width:70px">&nbsp;</div>
    <div class="suggestion shimmer" style="width:55px">&nbsp;</div>
    <div class="suggestion shimmer" style="width:65px">&nbsp;</div>`;
}

async function fetchPredictions() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) {
    renderSuggestions([], []);
    return;
  }

  showLoadingSuggestions();

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, client_id: selectedClient })
    });
    const data = await res.json();
    renderSuggestions(data.global_words || [], data.personal_words || []);
  } catch(e) {
    renderSuggestions(['...'], []);
  }
}

function renderSuggestions(globalWords, personalWords) {
  const bar = document.getElementById('suggestionBar');
  bar.innerHTML = '';

  if (!globalWords.length && !personalWords.length) {
    bar.innerHTML = '<span class="suggestion-label" style="color:var(--text2)">Type to predict…</span>';
    return;
  }

  // Global suggestions (purple)
  globalWords.forEach(w => {
    const btn = document.createElement('button');
    btn.className = 'suggestion global';
    btn.textContent = w;
    btn.title = 'Global model';
    btn.onclick = () => appendWord(w);
    bar.appendChild(btn);
  });

  // Divider if both
  if (personalWords.length) {
    const div = document.createElement('span');
    div.style.cssText = 'width:1px;background:var(--border);height:20px;flex-shrink:0;margin:0 2px';
    bar.appendChild(div);

    // Personal suggestions (green)
    personalWords.forEach(w => {
      const btn = document.createElement('button');
      btn.className = 'suggestion personal';
      btn.textContent = w;
      btn.title = `Client ${selectedClient} personalized`;
      btn.onclick = () => appendWord(w);
      bar.appendChild(btn);
    });
  }
}

function appendWord(word) {
  const inp = document.getElementById('textInput');
  const current = inp.value.trimEnd();
  inp.value = current + (current ? ' ' : '') + word + ' ';
  inp.focus();
  triggerPrediction();
}

// ── Send message ──────────────────────────────────────────────
function sendMessage() {
  const inp = document.getElementById('textInput');
  const text = inp.value.trim();
  if (!text) return;

  addBubble(text, 'sent',
    selectedClient === null ? 'You · Global' : `You · Client ${selectedClient}`);

  inp.value = '';
  renderSuggestions([], []);

  // Echo with "reply"
  setTimeout(() => {
    const label = selectedClient === null
      ? 'Global Model'
      : `Client ${selectedClient} · Personalized`;
    addBubble('✓ Message sent via ' + label, 'recv', 'System');
  }, 400);
}

function addBubble(text, type, label) {
  const msgs = document.getElementById('messages');
  const b = document.createElement('div');
  b.className = `bubble ${type}`;
  b.innerHTML = `<div class="label">${label}</div>${escapeHtml(text)}`;
  msgs.appendChild(b);
  msgs.scrollTop = msgs.scrollHeight;
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Input listener ────────────────────────────────────────────
document.getElementById('textInput').addEventListener('input', triggerPrediction);
document.getElementById('textInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});

// ── Init ──────────────────────────────────────────────────────
buildKeyboard();
buildClientGrid();
</script>
</body>
</html>
"""

# ── API routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data      = request.get_json()
    text      = data.get("text", "").strip()
    client_id = data.get("client_id")   # None = global only

    if not text:
        return jsonify(global_words=[], personal_words=[])

    # Global model: top-3 suggestions
    global_words = get_top_words(global_model, text, n=3)

    # Personalized model (if client selected and model exists)
    personal_words = []
    if client_id is not None and client_id in client_models:
        personal_words = get_top_words(client_models[client_id], text, n=3)

    return jsonify(global_words=global_words, personal_words=personal_words)


def get_top_words(model, text, n=3):
    """Return top-n predicted next words (skipping OOV)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from model import SEQ_LEN, VOCAB_SIZE

    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    token_ids = tokenizer.texts_to_sequences([text.lower()])[0]
    padded    = pad_sequences([token_ids], maxlen=SEQ_LEN,
                               padding="pre", truncating="pre")
    probs = model.predict(padded, verbose=0)[0]

    # Get top-10 candidates, filter OOV
    top_indices = np.argsort(probs)[-10:][::-1]
    words = []
    for idx in top_indices:
        word = index_to_word.get(idx, "")
        if word and word != "<OOV>" and len(word) > 1:
            words.append(word)
        if len(words) == n:
            break

    return words


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0")
