"""
STAGE 2 — Supervised classifier trained on the pseudo-labels.

Two backends behind one interface:
  * "deberta"  : fine-tunes microsoft/deberta-v3-small (SPEC-COMPLIANT default,
                 weighted cross-entropy for class imbalance).
  * "baseline" : TF-IDF + LogisticRegression(class_weight='balanced').
                 A safety net that runs anywhere (offline / no GPU). It does
                 NOT satisfy the "fine-tuned model" requirement and is only a
                 fallback for environments without transformers.

Inputs always include text + structured metadata (channel, type, resolution
hours) via `serialize_input`, satisfying the metadata requirement.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Input serialization (text + structured metadata -> one string)
# --------------------------------------------------------------------------- #
_RT_BAND = {12: "fast", 37: "moderate", 70: "slow", 95: "very_slow", -1: "unknown"}


def resolution_band(row) -> str:
    """Categorical resolution-time band (aligned with the Stage-1 percentile)."""
    pct = row.get("rt_pct", None)
    if pct is not None and pct in _RT_BAND:
        return _RT_BAND[pct]
    hrs = row.get("resolution_hours")          # fallback if not pseudo-labeled yet
    if pd.isna(hrs):
        return "unknown"
    return "slow" if float(hrs) >= 48 else "moderate"


def serialize_input(row) -> str:
    # assigned_priority is a legitimate audit-time INPUT: the mismatch label is
    # defined relative to it, so the model must see it to judge a conflict.
    # resolution is given as a categorical band (the raw hour count is not a
    # meaningful token to a text model). The text carries the severity signal
    # the model learns to read. Together: text + 2 structured metadata features.
    return (f"assigned_priority: {row.get('priority','')} | "
            f"channel: {row.get('channel','')} | "
            f"type: {row.get('ticket_type','')} | "
            f"resolution_speed: {resolution_band(row)} | "
            f"subject: {row.get('subject','')} | "
            f"description: {row.get('description','')}").strip()


def class_weights(y) -> np.ndarray:
    y = np.asarray(y)
    n = len(y)
    counts = np.array([(y == 0).sum(), (y == 1).sum()], dtype=float)
    counts[counts == 0] = 1.0
    return n / (2.0 * counts)        # inverse-frequency weighting


def transformers_available() -> bool:
    try:
        import torch, transformers, sentencepiece   # noqa
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# DeBERTa backend
# --------------------------------------------------------------------------- #
def _train_deberta(train_df, val_df, outdir, epochs, lr, batch_size, max_len):
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)

    tok = AutoTokenizer.from_pretrained(C.HF_MODEL_NAME)

    def to_ds(df):
        d = Dataset.from_dict({
            "text": [serialize_input(r) for _, r in df.iterrows()],
            "label": df["mismatch"].astype(int).tolist(),
        })
        return d.map(lambda b: tok(b["text"], truncation=True,
                                   max_length=max_len), batched=True)

    train_ds, val_ds = to_ds(train_df), to_ds(val_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.HF_MODEL_NAME, num_labels=2)

    cw = torch.tensor(class_weights(train_df["mismatch"]), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            logits = out.logits.float()
            w = cw.to(device=logits.device, dtype=logits.dtype)
            loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(outdir / "_hf"),
        num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=50, seed=C.SEED,
        fp16=torch.cuda.is_available(), report_to=[],
    )
    # transformers >=5 renamed `tokenizer` -> `processing_class`
    try:
        trainer = WeightedTrainer(model=model, args=args,
                                  train_dataset=train_ds, eval_dataset=val_ds,
                                  processing_class=tok)
    except TypeError:
        trainer = WeightedTrainer(model=model, args=args,
                                  train_dataset=train_ds, eval_dataset=val_ds,
                                  tokenizer=tok)
    trainer.train()
    (outdir / "model").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")


def _predict_deberta(df, model_dir, max_len, batch_size=32):
    import torch
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    texts = [serialize_input(r) for _, r in df.iterrows()]
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt").to(dev)
            p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            probs.append(p)
    return np.vstack(probs)


# --------------------------------------------------------------------------- #
# Baseline backend (sklearn)
# --------------------------------------------------------------------------- #
# Baseline backend (sklearn, NONLINEAR) — runs anywhere, no GPU/download.
# Uses an MLP because the mismatch label is an INTERACTION (assigned priority
# x inferred text severity) that a linear model cannot represent.
def _train_baseline(train_df, outdir):
    import joblib
    import pandas as pd
    from sklearn.utils import resample
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline

    # MLPClassifier has no class_weight -> oversample minority (imbalance fix)
    maj = train_df[train_df["mismatch"] == 0]
    mino = train_df[train_df["mismatch"] == 1]
    if 0 < len(mino) < len(maj):
        mino = resample(mino, replace=True, n_samples=len(maj),
                        random_state=C.SEED)
    bal = pd.concat([maj, mino]).sample(frac=1.0, random_state=C.SEED)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                              early_stopping=True, n_iter_no_change=6,
                              random_state=C.SEED)),
    ])
    X = [serialize_input(r) for _, r in bal.iterrows()]
    pipe.fit(X, bal["mismatch"].astype(int))
    joblib.dump(pipe, outdir / "baseline.joblib")


def _predict_baseline(df, outdir):
    import joblib
    pipe = joblib.load(outdir / "baseline.joblib")
    X = [serialize_input(r) for _, r in df.iterrows()]
    return pipe.predict_proba(X)


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def train_classifier(train_df, val_df, outdir, backend=None,
                     epochs=C.EPOCHS, lr=C.LR, batch_size=C.BATCH_SIZE,
                     max_len=C.MAX_LEN):
    backend = backend or C.MODEL_BACKEND
    if backend == "deberta" and not transformers_available():
        print("[model] transformers unavailable -> falling back to baseline. "
              "Install torch+transformers+sentencepiece for the spec model.")
        backend = "baseline"
    if backend == "deberta":
        _train_deberta(train_df, val_df, outdir, epochs, lr, batch_size, max_len)
    else:
        _train_baseline(train_df, outdir)
    return backend


def predict_proba(df, outdir, backend, max_len=C.MAX_LEN):
    if backend == "deberta":
        return _predict_deberta(df, outdir / "model", max_len)
    return _predict_baseline(df, outdir)


def proba_to_pred(probs, threshold=0.5):
    preds = (probs[:, 1] >= threshold).astype(int)
    conf = probs.max(axis=1)
    return preds, conf
