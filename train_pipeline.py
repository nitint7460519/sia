"""
SIA — standalone training pipeline.

    python train_pipeline.py --data data/sample_tickets.csv
    python train_pipeline.py --data data/real.csv --backend deberta --epochs 4
    python train_pipeline.py --data data/sample_tickets.csv --ablation

Produces in artifacts/:
    model/ (or baseline.joblib), scalers.json, metrics.json,
    agreement.json, ablation.csv (if --ablation), pseudo_labels.csv
"""
from __future__ import annotations
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C
from src import data as D
from src import signals as S
from src import model as M
from src import metrics as MET


def stratified_split(df):
    train, temp = train_test_split(
        df, test_size=C.TEST_SIZE + C.VAL_SIZE,
        stratify=df["mismatch"], random_state=C.SEED)
    rel = C.TEST_SIZE / (C.TEST_SIZE + C.VAL_SIZE)
    val, test = train_test_split(
        temp, test_size=rel, stratify=temp["mismatch"], random_state=C.SEED)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def run_ablation(df, outdir):
    """Each signal alone + pairs + full; report mismatch rate + baseline F1."""
    emb_ok = S.EmbeddingScorer().load().available
    combos = [("lex",), ("rt",), ("emb",), ("lex", "rt"),
              ("lex", "emb"), ("rt", "emb"), ("lex", "rt", "emb")]
    if not emb_ok:
        combos = [c for c in combos if "emb" not in c]
        print("[ablation] embedding unavailable -> reporting lex/rt combos only")
    rows = []
    for combo in combos:
        pl, _, _, _ = S.generate_pseudo_labels(df, signals=combo)
        rate = float(pl["mismatch"].mean())
        # quick TF-IDF F1 proxy so the table reflects learnability cheaply
        f1 = _quick_f1(pl)
        rows.append({"signals": "+".join(combo),
                     "mismatch_rate": round(rate, 4),
                     "proxy_macro_f1": round(f1, 4)})
    abl = pd.DataFrame(rows)
    abl.to_csv(outdir / "ablation.csv", index=False)
    print("\n=== ABLATION ===\n", abl.to_string(index=False))
    return abl


def _quick_f1(pl):
    import pandas as pd
    from sklearn.utils import resample
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    if pl["mismatch"].nunique() < 2:
        return 0.0
    tr, te = train_test_split(pl, test_size=0.25,
                              stratify=pl["mismatch"], random_state=C.SEED)
    # match the real trainer: oversample the minority class
    maj, mino = tr[tr.mismatch == 0], tr[tr.mismatch == 1]
    if 0 < len(mino) < len(maj):
        mino = resample(mino, replace=True, n_samples=len(maj), random_state=C.SEED)
    bal = pd.concat([maj, mino]).sample(frac=1.0, random_state=C.SEED)
    pipe = Pipeline([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                     ("c", MLPClassifier(hidden_layer_sizes=(128, 64),
                                         max_iter=300, random_state=C.SEED))])
    pipe.fit([M.serialize_input(r) for _, r in bal.iterrows()], bal["mismatch"])
    from sklearn.metrics import f1_score
    Xte = [M.serialize_input(r) for _, r in te.iterrows()]
    return f1_score(te["mismatch"], pipe.predict(Xte), average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", default=None, choices=[None, "deberta", "baseline"])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--no_embedding", action="store_true",
                    help="skip the semantic signal (offline-friendly)")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--outdir", default=str(C.ARTIFACTS_DIR))
    a = ap.parse_args()

    from pathlib import Path
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading & preparing {a.data}")
    df = D.load_prepared(a.data)
    print(f"      {len(df)} tickets | resolution known: {df.resolution_known.mean():.1%}")

    sig = ("lex", "rt") if a.no_embedding else ("lex", "rt", "emb")
    print(f"[2/5] Generating pseudo-labels (signals={sig})")
    pl, rt_scaler, embedder, calibrator = S.generate_pseudo_labels(df, signals=sig)
    used_emb = embedder is not None and embedder.available
    print(f"      mismatch rate: {pl.mismatch.mean():.1%} | "
          f"types: {pl[pl.mismatch==1].mismatch_type.value_counts().to_dict()}")

    # signal-agreement metric
    agree = S.signal_agreement(pl, "sev_lex",
                               "sev_emb" if used_emb else "sev_rt")
    json.dump(agree, open(outdir / "agreement.json", "w"), indent=2)
    print(f"      agreement: {agree}")

    if a.ablation:
        run_ablation(df, outdir)

    pl.to_csv(outdir / "pseudo_labels.csv", index=False)

    print("[3/5] Splitting (stratified train/val/test)")
    train, val, test = stratified_split(pl)
    print(f"      train={len(train)} val={len(val)} test={len(test)}")

    print(f"[4/5] Training classifier (backend={a.backend or C.MODEL_BACKEND})")
    backend = M.train_classifier(train, val, outdir, backend=a.backend,
                                 epochs=a.epochs)

    # persist everything inference needs
    scalers = {"backend": backend, "signals": list(sig),
               "used_embedding": bool(used_emb),
               "rt_scaler": rt_scaler.to_dict(),
               "calibrator": calibrator.to_dict()}
    json.dump(scalers, open(outdir / "scalers.json", "w"), indent=2)

    print("[5/5] Evaluating on held-out test split")
    probs = M.predict_proba(test, outdir, backend)
    preds, _ = M.proba_to_pred(probs)
    m = MET.evaluate(test["mismatch"], preds)
    json.dump(m, open(outdir / "metrics.json", "w"), indent=2)

    print("\n" + MET.report(test["mismatch"], preds))
    print("=== METRICS (held-out) ===")
    for k in ("accuracy", "macro_f1", "recall_consistent",
              "recall_mismatch", "min_per_class_recall"):
        print(f"  {k:22s}: {m[k]:.4f}")
    print("=== THRESHOLD CHECK ===")
    for k, v in m["passes"].items():
        print(f"  {k:22s}: {'PASS' if v else 'FAIL'}")
    if backend == "baseline":
        print("\n  NOTE: 'baseline' is the TF-IDF+MLP fallback (runs without a GPU). "
              "It is a nonlinear model and can meet the thresholds, but the spec "
              "asks for a fine-tuned transformer. Use --backend deberta on a GPU "
              "(Colab/Kaggle) for the official submission model.")


if __name__ == "__main__":
    main()
