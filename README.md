# CircadianAI — Sleep & Circadian Rhythm Analysis

> **AI-powered, browser-native sleep analysis using a Temporal Convolutional Network (TCN) — no server required.**

CircadianAI is a full-stack web application that analyses 7 days of personal sleep data and produces clinically-framed insights including insomnia risk, predicted sleep duration, recovery timeline, and a personalised adaptation strategy — all computed on-device via an ONNX model.

---

## ✨ Features

- **Edge AI inference** — ONNX model runs entirely in the browser via ONNX Runtime Web; zero latency, zero data sent to a server
- **Multi-task TCN** — single model produces 5 outputs simultaneously (duration, insomnia risk, recovery days, 7-day trajectory, adaptation strategy)
- **Event-aware predictions** — jet lag and night-shift context adjusts the model's features for disrupted circadian scenarios
- **Validated clinical scoring** — ISI, PHQ-9, GAD-7, MEQ questionnaires embedded and scored automatically
- **Supabase authentication** — secure sign-up/login and persistent session history stored in a Postgres DB
- **HRV & wearable input** — supports manual entry or CSV paste of RMSSD/SDNN from any wearable device
- **7-day trend charts** — sleep duration, insomnia risk, HRV, and efficiency history visualised via Chart.js
- **Offline fallback** — heuristic scoring works even when ONNX model or database is unavailable

---

## 🖥 Demo Flow

```
login.html  →  index.html (7-day data entry)  →  questionnaire.html (clinical scores)  →  results.html
```

1. Sign up or log in via Supabase Auth
2. Enter 7 consecutive nights of sleep data (duration, efficiency, HR, HRV, steps, light)
3. Complete the clinical questionnaire (ISI, PHQ-9, GAD-7, MEQ) — or use the interactive quiz
4. Add optional event context (jet lag, night shift, timezone shift)
5. View your personalised AI report on the results page

---

## 🗂 Project Structure

```
circadianai/
│
├── index.html                  # Day-by-day sleep data entry (Step 1)
├── login.html                  # Supabase Auth — sign in / sign up
├── questionnaire.html          # Clinical scoring + ONNX inference (Step 2)
├── results.html                # Results dashboard (Step 3)
│
├── circadian_edge.onnx         # Quantised edge model for browser inference (303 KB)
├── circadian_edge_meta.json    # Model metadata (feature names, output schema)
│
├── tcn_model.py                # CircadianTCN + EdgeTCN architecture (PyTorch)
├── train_tcn_model.py          # Multi-task training script (5 heads)
├── preprocess.py               # Feature engineering pipeline (19 features)
├── inference.py                # Python inference wrapper (for backend/testing)
├── export_onnx.py              # PyTorch → ONNX export utility
├── run.py                      # Quick CLI test runner
│
├── final_tcn_dataset.csv       # Synthetic training dataset
├── splits.npz                  # Pre-computed train/val/test splits
├── tcn_model.pth               # Full model checkpoint
├── tcn_edge.pt                 # TorchScript edge model (pre-ONNX)
│
└── requirements.txt            # Python dependencies
```

---

## 🤖 Model Architecture — CircadianTCN v3

The TCN (Temporal Convolutional Network) processes a 7-day sequence of 19 features and produces 5 simultaneous outputs via separate task heads.

### Input

| Dimension | Value | Description |
|-----------|-------|-------------|
| Batch | B | number of samples |
| Sequence | 7 | consecutive nights |
| Features | 19 | per-night measurements |

**19 input features:**

| # | Feature | Source |
|---|---------|--------|
| 0 | `sleep_duration_hrs` | wearable / manual |
| 1 | `sleep_efficiency` | wearable / manual |
| 2 | `heart_rate_norm` | resting HR normalised |
| 3 | `rmssd_norm` | HRV metric |
| 4 | `sdnn_norm` | HRV metric |
| 5 | `steps_norm` | daily activity |
| 6 | `light_exposure_norm` | lux reading |
| 7 | `bedtime_sin` | circular encoding |
| 8 | `bedtime_cos` | circular encoding |
| 9 | `isi_norm` | Insomnia Severity Index |
| 10 | `phq_norm` | PHQ-9 depression screen |
| 11 | `gad_norm` | GAD-7 anxiety screen |
| 12 | `meq_norm` | Morningness-Eveningness |
| 13 | `age_norm` | user age (16–80) |
| 14 | `event_jetlag` | binary: recent air travel |
| 15 | `event_nightshift` | binary: shift work |
| 16 | `tz_shift_norm` | timezone offset (±12h) |
| 17 | `shift_week_norm` | shift cycle week (0–4) |
| 18 | `days_since_event_norm` | days since disruption event |

### Architecture

```
Input (B, 7, 19)
    │
    ▼
Input Projection  Conv1d(19 → 64)
    │
    ▼
TCN Blocks ×4     CausalConv + LayerNorm + GELU + Dropout
  dilation: 1, 2, 4, 8
    │
    ▼
Multi-Head Attention  (circadian phase attention, 4 heads)
    │
    ▼
Global Average Pool → (B, 64)
    │
Event Gate ←── last-day event features (5)
    │
    ▼
Shared FC  64 → 128 → 64
    │
    ├──► Duration Head      → sleep hours  [0–12]
    ├──► Insomnia Head      → risk prob    [0–1]
    ├──► Recovery Head      → days         [0–21]
    ├──► Trajectory Head    → 7-day risk   [0–1] ×7
    └──► Strategy Head      → logits       ×5
```

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `sleep_duration` | float | 0–12 hrs | predicted sleep tonight |
| `insomnia_prob` | float | 0–1 | insomnia probability |
| `recovery_days` | float | 0–21 days | estimated time to recovery |
| `insomnia_trajectory` | float[7] | 0–1 per day | 7-day risk forecast |
| `strategy_logits` | float[5] | raw logits | adaptation strategy |

**Adaptation strategies (argmax of softmax):**

| ID | Strategy |
|----|----------|
| 0 | Maintain current routine |
| 1 | Morning bright light therapy |
| 2 | Melatonin + gradual phase advance |
| 3 | Sleep restriction therapy |
| 4 | Gradual shift schedule adaptation |

### Training

Multi-task loss with per-task weights:

| Task | Loss Function | Weight |
|------|---------------|--------|
| Duration | Huber | 0.20 |
| Insomnia | Focal BCE | 0.25 |
| Recovery | Huber | 0.20 |
| Trajectory | MSE (7-day) | 0.20 |
| Strategy | CrossEntropy | 0.15 |

---

## 🚀 Getting Started

### Option A — Run locally (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/circadianai.git
cd circadianai

# 2. Serve with any static file server — do NOT open via file://
python -m http.server 8080

# 3. Open in browser
# http://localhost:8080/login.html
```

> **Important:** The ONNX runtime and Supabase SDK require HTTP, not `file://`. Always use a local server.

### Option B — GitHub Pages

Push to a GitHub repo, enable Pages from the `main` branch, and the app is live immediately. No build step required — it's all static HTML/CSS/JS.

---

## 🗄 Supabase Setup (Required for Auth + History)

The app uses Supabase for user authentication and saving session history. Without this step, the app still works for single-use analysis via `sessionStorage`, but results won't be saved.

### 1. Create a Supabase project

Go to [supabase.com](https://supabase.com) → New Project. Note your **Project URL** and **anon public key**.

### 2. Update credentials in all four HTML files

Search for `SUPABASE_URL` and `SUPABASE_KEY` in `login.html`, `index.html`, `questionnaire.html`, and `results.html` and replace with your project values:

```javascript
const SUPABASE_URL = 'https://your-project-ref.supabase.co';
const SUPABASE_KEY = 'your-anon-public-key';
```

### 3. Run the database migration

In your Supabase dashboard → **SQL Editor**, paste and run:

```sql
-- Create the sessions table
CREATE TABLE public.sleep_sessions (
  id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                  UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
  is_draft                 BOOLEAN NOT NULL DEFAULT false,

  -- Clinical scores
  isi_score                INTEGER,
  phq_score                INTEGER,
  gad_score                INTEGER,
  meq_score                INTEGER,

  -- Core predictions
  predicted_sleep_hrs      NUMERIC(5,2),
  sleep_duration_label     TEXT,
  recommended_sleep_hrs    NUMERIC(5,2),
  insomnia_probability     NUMERIC(8,4),
  insomnia_risk_level      TEXT,
  circadian_bedtime_est    NUMERIC(5,2),

  -- 7-day trends
  avg_sleep_duration_hrs   NUMERIC(5,2),
  avg_sleep_efficiency_pct NUMERIC(5,1),
  avg_hrv_rmssd_ms         NUMERIC(6,1),
  duration_7day            TEXT,
  efficiency_7day          TEXT,
  hrv_7day                 TEXT,

  -- Insights
  insights                 TEXT,
  recommendations          TEXT,

  -- v3 event context
  event_type               TEXT,
  tz_shift_hrs             NUMERIC(5,2),
  recovery_days_predicted  NUMERIC(5,1),
  adaptation_strategy      TEXT,
  adaptation_strategy_id   INTEGER,
  insomnia_trajectory      TEXT
);

-- Enable Row Level Security
ALTER TABLE public.sleep_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own sessions"   ON public.sleep_sessions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own sessions" ON public.sleep_sessions FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own sessions" ON public.sleep_sessions FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own sessions" ON public.sleep_sessions FOR DELETE USING (auth.uid() = user_id);

-- Indexes
CREATE INDEX idx_sleep_sessions_user_id    ON public.sleep_sessions (user_id);
CREATE INDEX idx_sleep_sessions_created_at ON public.sleep_sessions (created_at DESC);
```

### 4. Configure Auth settings

In Supabase → **Authentication → URL Configuration**:

- Set **Site URL** to your deployment URL (e.g. `http://localhost:8080` or `https://your-username.github.io/circadianai`)
- Add the same URL to **Redirect URLs**
- Under **Email** provider → disable **Confirm email** for easier local testing

---

## 🐍 Python Backend (Optional — Training & Inference)

The Python files are for training the model and running server-side inference. They are not required to run the web app.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train_tcn_model.py
```

Outputs: `tcn_model.pth` (full checkpoint) and `tcn_edge.pt` (TorchScript edge model).

### Export to ONNX

```bash
python export_onnx.py
```

Outputs: `circadian_edge.onnx` and `circadian_edge_meta.json`. Place both files in the same directory as the HTML files.

### Run a quick inference test

```bash
python run.py
```

---

## ⚠️ Clinical Disclaimer

> **This is a screening tool only.** CircadianAI is built on a machine learning model trained on synthetic data. It is **not a medical diagnostic device** and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified sleep physician, chronobiologist, or healthcare professional before making clinical decisions based on this output.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Vanilla HTML5, CSS3, JavaScript (ES2022) |
| AI Inference | ONNX Runtime Web (browser) |
| Model Training | PyTorch 2.x, multi-task TCN |
| Charts | Chart.js |
| Auth & DB | Supabase (Postgres + Auth) |
| Hosting | GitHub Pages (static) or any HTTP server |

---

## 📋 Version History

| Version | Key Changes |
|---------|-------------|
| v1.0 | Basic heuristic sleep scoring, no ML |
| v2.0 | TCN model, 13 features, 2 outputs (duration + insomnia) |
| v3.0 | 19 features, 5 outputs, event context, Supabase auth, full results dashboard |

---

## 📄 License

MIT License — see `LICENSE` for details.

---

*Built as a Track A AI Development project · CircadianAI TCN · Sleep Analysis · v3.0*
