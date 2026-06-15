"""
SIA — Streamlit web app.

    streamlit run app.py

Loads the trained model + fitted scalers from artifacts/. Supports single-ticket
form input and batch CSV upload, returns a binary judgment + Evidence Dossier,
and renders a Priority Mismatch Dashboard with a severity-delta heatmap.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from src import config as C
from src import data as D
from src import signals as S
from src import model as M
from src import dossier as DOSS

st.set_page_config(page_title="Support Integrity Auditor", layout="wide")
ARTI = C.ARTIFACTS_DIR


# --------------------------------------------------------------------------- #
# Loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model & scalers…")
def load_artifacts():
    meta = json.load(open(ARTI / "scalers.json"))
    rt_scaler = S.RTScaler.from_dict(meta["rt_scaler"])
    calibrator = S.SeverityCalibrator.from_dict(meta.get("calibrator"))
    embedder = S.EmbeddingScorer().load() if meta.get("used_embedding") else None
    return meta, rt_scaler, embedder, calibrator


def score_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    meta, rt_scaler, embedder, calibrator = load_artifacts()
    df = D.prepare(df_raw)
    pl, _, _, _ = S.generate_pseudo_labels(
        df, rt_scaler=rt_scaler, embedder=embedder,
        calibrator=calibrator, signals=tuple(meta["signals"]))
    probs = M.predict_proba(pl, ARTI, meta["backend"])
    preds, conf = M.proba_to_pred(probs)
    pl["prediction"] = preds
    pl["judgment"] = ["Mismatch" if p else "Consistent" for p in preds]
    pl["confidence"] = conf.round(3)
    return pl


def dossier_for(row) -> dict:
    d = DOSS.build_dossier(row, confidence=row["confidence"])
    viol = DOSS.verify_dossier(d, row)
    if viol:
        d["_violations"] = viol
    return d


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🛡️ Support Integrity Auditor (SIA)")
st.caption("Detects Priority Mismatch — tickets whose objective severity "
           "conflicts with their human-assigned priority. Every flag ships "
           "with a grounded, hallucination-checked Evidence Dossier.")

if not (ARTI / "scalers.json").exists():
    st.error("No trained model found in `artifacts/`. Run "
             "`python train_pipeline.py --data data/sample_tickets.csv` first.")
    st.stop()

mode = st.radio("Input mode", ["Single ticket", "Batch CSV upload"],
                horizontal=True)

PRIORITIES = ["Low", "Medium", "High", "Critical"]
CHANNELS = ["Email", "Chat", "Phone", "Social media"]

# --------------------------------------------------------------------------- #
# Single ticket
# --------------------------------------------------------------------------- #
if mode == "Single ticket":
    c1, c2 = st.columns(2)
    with c1:
        subject = st.text_input("Ticket subject", "Urgent: cannot access account")
        description = st.text_area(
            "Ticket description",
            "The whole system is down and we cannot access anything. "
            "This is blocking our entire team.", height=140)
    with c2:
        priority = st.selectbox("Assigned priority", PRIORITIES, index=0)
        channel = st.selectbox("Channel", CHANNELS, index=0)
        ttype = st.text_input("Ticket type", "Technical issue")
        res_hours = st.number_input("Resolution time (hours)", 0.0, 5000.0, 72.0)

    if st.button("Audit ticket", type="primary"):
        row = pd.DataFrame([{
            "Ticket ID": "SINGLE-001", "Ticket Subject": subject,
            "Ticket Description": description, "Ticket Priority": priority,
            "Ticket Channel": channel, "Ticket Type": ttype,
            "Resolution Time": res_hours,
        }])
        pl = score_frame(row)
        r = pl.iloc[0]
        verdict = r["judgment"]
        if verdict == "Mismatch":
            st.error(f"⚠️ PRIORITY MISMATCH — {r['mismatch_type']} "
                     f"(confidence {r['confidence']:.0%})")
            st.json(dossier_for(r))
        else:
            st.success(f"✅ CONSISTENT — assigned priority matches inferred "
                       f"severity '{r['inferred_sev']}' "
                       f"(confidence {r['confidence']:.0%})")

# --------------------------------------------------------------------------- #
# Batch CSV
# --------------------------------------------------------------------------- #
else:
    up = st.file_uploader("Upload a tickets CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        with st.spinner("Auditing tickets…"):
            pl = score_frame(raw)

        n, n_mm = len(pl), int(pl["prediction"].sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Tickets audited", n)
        k2.metric("Flagged mismatches", n_mm)
        k3.metric("Mismatch rate", f"{n_mm/n:.1%}" if n else "—")

        st.subheader("Priority Mismatch Dashboard")
        d1, d2 = st.columns(2)
        with d1:
            vc = pl["judgment"].value_counts().reset_index()
            vc.columns = ["judgment", "count"]
            st.plotly_chart(px.bar(vc, x="judgment", y="count",
                                   color="judgment", title="Judgment distribution"),
                            use_container_width=True)
        with d2:
            mm = pl[pl["prediction"] == 1]
            if len(mm):
                tc = mm["mismatch_type"].value_counts().reset_index()
                tc.columns = ["mismatch_type", "count"]
                st.plotly_chart(px.pie(tc, names="mismatch_type", values="count",
                                       title="Mismatch types"),
                                use_container_width=True)

        # top contributing signals (counted from grounded evidence)
        sig_counts = {}
        dossiers = []
        for _, r in pl[pl["prediction"] == 1].iterrows():
            d = dossier_for(r)
            dossiers.append(d)
            for ev in d["feature_evidence"]:
                sig_counts[ev["signal"]] = sig_counts.get(ev["signal"], 0) + 1
        if sig_counts:
            sc = pd.DataFrame({"signal": list(sig_counts),
                               "count": list(sig_counts.values())})
            st.plotly_chart(px.bar(sc, x="signal", y="count",
                                   title="Top contributing signals (flagged tickets)"),
                            use_container_width=True)

        # severity-delta heatmap: type x channel
        st.subheader("Severity-delta heatmap (mean |Δ|)")
        piv = pl.pivot_table(index="ticket_type", columns="channel",
                             values="delta",
                             aggfunc=lambda s: s.abs().mean())
        if piv.size:
            st.plotly_chart(px.imshow(piv, text_auto=".1f", aspect="auto",
                                      color_continuous_scale="Reds",
                                      labels=dict(color="mean |Δ|")),
                            use_container_width=True)

        st.subheader("Results")
        show = ["ticket_id", "priority", "inferred_sev", "judgment",
                "mismatch_type", "delta", "confidence"]
        st.dataframe(pl[[c for c in show if c in pl.columns]],
                     use_container_width=True, height=300)

        st.subheader("Evidence Dossiers (flagged tickets)")
        st.caption("Every field is traceable to a ticket column — verified "
                   "hallucination-free.")
        for d in dossiers[:50]:
            with st.expander(f"{d['ticket_id']} — {d['mismatch_type']} "
                             f"(Δ={d['severity_delta']})"):
                st.json(d)
        st.download_button("Download all dossiers (JSON)",
                           json.dumps(dossiers, indent=2),
                           "dossiers.json", "application/json")
