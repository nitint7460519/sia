# 🛡️ Support Integrity Auditor (SIA)

**Detecting Priority Mismatch in customer-support tickets — without ever being given a mismatch label.**

SIA audits support tickets to find cases where a ticket's *objective severity* (what the
text and operational signals imply) conflicts with the *priority a human assigned*. It flags
two failure modes and explains every flag with a grounded, hallucination-checked dossier:

- **Hidden Crisis** — a genuinely severe ticket assigned a low priority (inferred ≫ assigned).
- **False Alarm** — a trivial ticket assigned a high priority (inferred ≪ assigned).

---

## The core problem

The dataset (20,000 tickets, columns `Priority_Level`, `Issue_Category`, `Ticket_Subject`,
`Ticket_Description`, `Resolution_Time_Hours`, `Ticket_Channel`, …) ships an assigned
`Priority_Level`, but **no ground-truth label for whether that priority is correct**. So this
is not ordinary supervised classification — the supervision has to be *manufactured* from the
data itself and then learned. SIA does exactly that with a mandatory three-stage pipeline.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                     RAW CRM TICKET                         │
                    │  subject · description · priority · channel · type · times │
                    └──────────────────────────────┬───────────────────────────┘
                                                    │
              ┌─────────────────────────────────────┴─────────────────────────────────┐
              │  STAGE 1 — SELF-SUPERVISED PSEUDO-LABELING  (src/signals.py)            │
              │                                                                         │
              │   s_lex  lexicon + negation ─┐                                          │
              │   s_rt   resolution-time pct ─┼─►  weighted fusion ─► inferred severity │
              │   s_emb  semantic embedding ─┘        (0..3 ordinal)                    │
              │                                                                         │
              │   mismatch = |inferred − assigned| ≥ 2                                  │
              │   type     = Hidden Crisis (Δ>0) | False Alarm (Δ<0)                    │
              └─────────────────────────────────────┬─────────────────────────────────┘
                                                    │  binary pseudo-labels
              ┌─────────────────────────────────────┴─────────────────────────────────┐
              │  STAGE 2 — SUPERVISED CLASSIFIER  (src/model.py)                        │
              │                                                                         │
              │   input  = text + structured metadata                                  │
              │            (assigned_priority, channel, type, resolution_band)         │
              │   model  = fine-tuned DeBERTa-v3-small  (weighted cross-entropy)        │
              │            └ fallback: TF-IDF + MLP (nonlinear, runs without a GPU)     │
              └─────────────────────────────────────┬─────────────────────────────────┘
                                                    │  judgment + confidence
              ┌─────────────────────────────────────┴─────────────────────────────────┐
              │  STAGE 3 — EVIDENCE DOSSIER  (src/dossier.py)                           │
              │   deterministic templating from real fields  ─►  verify_dossier()       │
              │   every evidence value is traceable to a column  ⇒  0 hallucinations    │
              └─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Self-supervised pseudo-labels

Three **independent** severity signals, each mapped to the same `0..3` ordinal scale, are
fused into a single inferred severity. Fusing ≥2 signals is required and also makes the label
robust: each signal fails in a different way, so the combination covers each one's blind spot.

| Signal | Source field | What it catches | Blind spot it covers |
|---|---|---|---|
| `s_lex` — lexicon + negation | `subject`,`description` | explicit escalation language ("outage", "urgent") | negation-aware ("**not** working" ≠ calm) |
| `s_rt` — resolution-time percentile | `Resolution_Time_Hours` | operational reality (long resolutions ⇒ hard tickets) | severity with no alarming words |
| `s_emb` — semantic embedding | `subject`,`description` | **paraphrased** urgency with *no keywords* | keyword-stuffing / keyword-free crises |

`s_emb` uses `all-MiniLM-L6-v2` and scores a ticket by its cosine similarity to urgent vs.
calm prototype sentences. This is the **adversarial-robustness** lever: a ticket like
*"the platform is unreachable and customers are furious"* contains none of the lexicon
keywords yet still scores as high severity.

**Fusion.** A weighted continuous score `0.40·s_lex + 0.25·s_rt + 0.35·s_emb` (weights
auto-renormalized if a signal is unavailable) is then **quantile-calibrated** onto the `0..3`
ordinal so that inferred severity is on the *same scale* as the assigned priority. This step
matters on real data: raw fused scores cluster in the middle, which — without calibration —
would make almost every High/Critical ticket look like a False Alarm purely by construction.
The calibrator (`SeverityCalibrator`) sets its cut points at the quantiles of the fused score
corresponding to the assigned-priority class proportions, and is fitted on the training split
and reused at inference. **Mismatch** is declared when `|inferred − assigned| ≥ 2` — a
two-level gap, deliberately conservative so borderline disagreements aren't flagged.

> Text is cleaned before scoring: the `Hi Support,` boilerplate greeting is stripped and the
> ticket is reduced to its **core issue sentence** (`subject` + first description sentence),
> discarding the trailing filler the dataset appends.

### Fusion justification (ablation + signal agreement)

Reproduce with `python train_pipeline.py --data data/tickets.csv --ablation`.
On the **real 20,000-ticket dataset**, comparing the deployed signals:

| Signals | Mismatch rate | Hidden Crisis | False Alarm | Held-out macro-F1 |
|---|---|---|---|---|
| lex | 0.194 | 1,341 | 2,529 | 0.975 |
| rt | 0.228 | 980 | 3,587 | 0.992 |
| **lex + rt** *(deployed)* | **0.243** | **1,941** | **2,913** | **0.906** |

A single signal is *individually* easier for a model to reproduce (a simpler labeling
function), but each is lopsided and brittle: `rt` alone barely detects Hidden Crises (980 vs
3,587), and `lex` alone misses operationally-hard tickets that contain no alarming words.
The **fusion produces the most balanced detection of both mismatch types** and combines
complementary text + operational evidence — the right trade for a tiny dip in raw learnability.
**Signal agreement** (`artifacts/agreement.json`): lex↔rt agreement **0.48**, Cohen's κ ≈ **0**
— the two signals are essentially *independent*, which is exactly why fusing them adds
information rather than echoing it. The optional `s_emb` semantic signal adds the
adversarial-robustness layer (keyword-free crises) and can be enabled with the full `--backend
deberta` run.

---

## Stage 2 — The classifier

The classifier learns to **reproduce the pseudo-labeling function and generalize it to unseen
wording**. Its input is text **plus structured metadata**, serialized as one string:

```
assigned_priority: critical | channel: Email | type: Technical issue |
resolution_speed: fast | subject: ... | description: ...
```

Two design points worth calling out:

1. **`assigned_priority` is an input feature, not leakage.** Mismatch is *defined relative to*
   the assigned priority, so the model must see it to judge a conflict. The label still requires
   inferring true severity from the text + resolution band, so the priority alone cannot
   determine the answer.
2. **Mismatch is an interaction (XOR-like).** `assigned=Critical ∧ text=trivial → mismatch`, but
   `assigned=Critical ∧ text=critical → consistent`. A linear model provably cannot represent
   this — in testing, TF-IDF + logistic regression collapsed (~0.55 acc), while a **nonlinear**
   model (DeBERTa, or the TF-IDF+MLP fallback) captures it cleanly. This is precisely why the
   spec mandates a fine-tuned model rather than a frozen zero-shot one.

**Class imbalance** (mismatches are the minority) is handled by **class-weighted cross-entropy**
in the DeBERTa path and **minority oversampling** in the MLP fallback.

### Two backends

| Backend | What it is | Use it for |
|---|---|---|
| `deberta` *(default, spec-compliant)* | fine-tuned `microsoft/deberta-v3-small`, weighted loss | the official submission; run on a GPU (Colab/Kaggle) |
| `baseline` | TF-IDF + `MLPClassifier(128,64)` + oversampling | offline / no-GPU demo; runs and passes anywhere |

The code auto-falls back to `baseline` if `torch`/`transformers` are missing, so the repo
**never hard-crashes**.

---

## Stage 3 — Evidence Dossier (zero hallucination)

Every flagged ticket produces a dossier in the exact required schema. It is built by
**deterministic templating from real field values** — the model never free-writes evidence —
and then passed through `verify_dossier()`, which fails if any evidence value cannot be traced
back to a concrete source field.

```json
{
  "ticket_id": "T000000",
  "assigned_priority": "Critical",
  "inferred_severity": "Low",
  "mismatch_type": "False Alarm",
  "severity_delta": -3,
  "feature_evidence": [
    { "signal": "resolution_time", "value": "5h",
      "interpretation": "12th percentile -> low severity" }
  ],
  "constraint_analysis": "Ticket was assigned 'critical' priority, but its text shows no escalation language and resolution took 5h (12th pct), implying a 'Low' severity. The 3-level gap indicates a False Alarm.",
  "confidence": 1.0
}
```

> **Hallucination guarantee.** Across every flagged ticket in testing, `verify_dossier()`
> reported **0 violations**. Keyword evidence must appear verbatim in the ticket text;
> resolution evidence requires a real measured duration; `severity_delta` must equal
> `inferred_ord − assigned_ord`. Nothing is asserted that the data does not support.

---

## Results

Held-out test split (3,000 tickets), **real 20,000-ticket Kaggle dataset**, `baseline`
backend (TF-IDF + MLP), signals = lex+rt:

| Metric | Threshold | Result |
|---|---|---|
| Accuracy | ≥ 0.83 | **0.941** |
| Macro-F1 | ≥ 0.82 | **0.917** |
| Per-class recall — Consistent | ≥ 0.78 | **0.973** |
| Per-class recall — Mismatch | ≥ 0.78 | **0.841** |
| Hallucinated dossiers | 0 | **0** |

All thresholds pass on real data with the no-GPU fallback model. The fine-tuned
DeBERTa-v3-small backend (`--backend deberta`, run on a GPU) targets the same or better,
plus the adversarial-robustness bonus. Numbers are reproduced into `artifacts/metrics.json`
by `train_pipeline.py`; train/val/test = 14,000 / 3,000 / 3,000 (stratified).

**Adversarial robustness (bonus).** The optional embedding signal flags keyword-free crises
(*"the platform is unreachable and customers are furious"*) and the negation handling defuses
keyword-stuffed false alarms (*"this is **not** urgent, just a question"*), targeting the
≥7/10 robustness criterion.

---

## Repository layout

```
sia/
├── README.md
├── requirements.txt          # full pinned deps (training + app)
├── requirements-app.txt      # slim deps for hosting the app (no torch)
├── notebook.ipynb            # end-to-end reproducible walkthrough
├── train_pipeline.py         # load → pseudo-label → split → train → evaluate
├── predict.py                # CSV in → predictions.csv + verified dossiers.json
├── app.py                    # Streamlit: form + batch + dashboard + heatmap
├── .streamlit/config.toml    # app theme + server config (deploy-ready)
├── artifacts/                # trained model + scalers + metrics (ship-ready)
└── src/
    ├── config.py             # lexicons, weights, thresholds, column maps, hyperparams
    ├── data.py               # robust column mapping + cleaning + resolution-time
    ├── signals.py            # Stage 1: signals, fusion, calibration, mismatch, agreement
    ├── model.py              # Stage 2: DeBERTa + MLP fallback, serialization, imbalance
    ├── dossier.py            # Stage 3: dossier builder + hallucination verifier
    └── metrics.py            # accuracy / macro-F1 / per-class recall / thresholds
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) Data — the real Kaggle CRM dataset (recommended)
#    https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data
#    save it to data/tickets.csv   (the loader maps its real column names automatically)

# 2) Train (no GPU needed — TF-IDF+MLP fallback, passes thresholds on the real data)
python train_pipeline.py --data data/tickets.csv --backend baseline --no_embedding --ablation

# 2b) Train the spec model on a GPU (Colab/Kaggle)
python train_pipeline.py --data data/tickets.csv --backend deberta --epochs 4

# 3) Predict on new tickets → predictions + grounded dossiers
python predict.py --input data/tickets.csv --out_pred predictions.csv --out_dossier dossiers.json

# 4) Launch the web app
streamlit run app.py
```

The loader maps the real Kaggle column names automatically (`Priority_Level`,
`Ticket_Description`, `Resolution_Time_Hours`, …), so the same commands work on the real CSV.

---

## Deployment — getting a hosted URL (Streamlit Community Cloud, free)

The repo ships the trained baseline model (`artifacts/scalers.json` +
`artifacts/baseline.joblib`, ~16 MB), so the hosted app works with **no GPU and no model
download**. End to end it takes ~3 minutes:

1. **Push to GitHub.** Make sure `app.py`, `src/`, `.streamlit/`, and `artifacts/scalers.json`
   + `artifacts/baseline.joblib` are committed (the provided `.gitignore` already keeps those
   two artifacts while ignoring the heavy ones).
2. **Go to https://share.streamlit.io** → *Create app* → pick your repo/branch.
3. **Main file path:** `app.py`. **Dependencies:** set the requirements file to
   `requirements-app.txt` (slim, torch-free — builds fast and fits the free tier).
4. Click **Deploy**. Streamlit gives you a public URL like
   `https://<your-app>.streamlit.app` — that is the hosted deliverable.

The live app then accepts single-ticket form input or a batch CSV upload, returns a binary
judgment + full Evidence Dossier per ticket, and renders the Priority Mismatch Dashboard
(judgment distribution, mismatch-type breakdown, top contributing signals) and the
severity-delta heatmap across ticket categories × channels.

> To host the fine-tuned **DeBERTa** model instead of the baseline, push the fine-tuned
> weights to the Hugging Face Hub, load them by ID in `src/model.py`, and use the full
> `requirements.txt`.

### Run locally

```bash
pip install -r requirements-app.txt   # or requirements.txt for the full pipeline
streamlit run app.py                  # opens http://localhost:8501
```

---

## Why this is reliable

- **No hard dependency on a GPU or network** — the fallback backend and graceful signal
  degradation keep every command runnable; the spec model is one flag away on a GPU.
- **Robust ingestion** — fuzzy column mapping, placeholder stripping, and resolution-time
  derivation survive messy real-world CSVs.
- **Provable grounding** — `verify_dossier()` turns "no hallucinations" from a promise into a
  test that runs on every flag.
- **Reproducible** — fixed seeds, a bundled data generator, and a one-command pipeline.
