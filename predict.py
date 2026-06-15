"""
SIA — inference.

    python predict.py --input new_tickets.csv \
                      --out_pred predictions.csv \
                      --out_dossier dossiers.json

Loads the trained model + fitted scalers from artifacts/, classifies each
ticket, and emits a grounded Evidence Dossier for every flagged mismatch.
Every dossier is verified for hallucinations before being written.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from src import config as C
from src import data as D
from src import signals as S
from src import model as M
from src import dossier as DOSS


def load_artifacts(outdir: Path):
    meta = json.load(open(outdir / "scalers.json"))
    rt_scaler = S.RTScaler.from_dict(meta["rt_scaler"])
    calibrator = S.SeverityCalibrator.from_dict(meta.get("calibrator"))
    embedder = None
    if meta.get("used_embedding"):
        embedder = S.EmbeddingScorer().load()
    return meta, rt_scaler, embedder, calibrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default=str(C.ARTIFACTS_DIR))
    ap.add_argument("--out_pred", default="predictions.csv")
    ap.add_argument("--out_dossier", default="dossiers.json")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    meta, rt_scaler, embedder, calibrator = load_artifacts(outdir)
    signals_used = tuple(meta["signals"])

    df = D.load_prepared(a.input)
    pl, _, _, _ = S.generate_pseudo_labels(
        df, rt_scaler=rt_scaler, embedder=embedder,
        calibrator=calibrator, signals=signals_used)

    probs = M.predict_proba(pl, outdir, meta["backend"])
    preds, conf = M.proba_to_pred(probs)
    pl["prediction"] = preds
    pl["judgment"] = ["Mismatch" if p else "Consistent" for p in preds]
    pl["confidence"] = conf.round(3)

    # predictions table
    cols = ["ticket_id", "priority", "inferred_sev", "judgment",
            "mismatch_type", "severity_delta" if "severity_delta" in pl else "delta",
            "confidence"]
    cols = [c for c in cols if c in pl.columns]
    pl[cols].to_csv(a.out_pred, index=False)

    # dossiers for flagged tickets, verified
    dossiers, hallucinated = [], 0
    for _, r in pl[pl["prediction"] == 1].iterrows():
        d = DOSS.build_dossier(r, confidence=r["confidence"])
        viol = DOSS.verify_dossier(d, r)
        if viol:
            hallucinated += 1
            d["_violations"] = viol
        dossiers.append(d)
    json.dump(dossiers, open(a.out_dossier, "w"), indent=2)

    print(f"Wrote {len(pl)} predictions -> {a.out_pred}")
    print(f"Flagged {len(dossiers)} mismatches -> {a.out_dossier}")
    print(f"Hallucination violations: {hallucinated} (target: 0)")


if __name__ == "__main__":
    main()
