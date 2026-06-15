"""
STAGE 1 — Self-supervised pseudo-label generation.

Three independent severity signals in [0, 3] are fused into an inferred
severity, which is compared to the human-assigned priority to derive a
binary mismatch label. No ground-truth labels are used anywhere.

    s_lex  : rule-based lexicon + negation        (keyword evidence)
    s_rt   : resolution-time percentile bucket     (structured evidence)
    s_emb  : semantic similarity to urgency protos (adversarial defense)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# Signal A — lexicon severity
# --------------------------------------------------------------------------- #
# Pre-split terms by phrase length so multi-word terms ("system down") match.
_TERMS = sorted(C.LEXICON.items(), key=lambda kv: -len(kv[0].split()))


def s_lex(text: str):
    """Return (severity 0..3, matched_keywords list, total_weight)."""
    t = " " + str(text).lower() + " "
    toks = t.split()
    score, hits = 0.0, []
    for term, w in _TERMS:
        if term in t:
            # negation only cancels positive (severity-raising) terms
            if w > 0:
                try:
                    first = term.split()[0]
                    idx = toks.index(first)
                    left = " ".join(toks[max(0, idx - C.LEX_NEG_WINDOW):idx])
                    if any(neg in left for neg in C.NEGATORS):
                        continue
                except ValueError:
                    pass
                hits.append(term)
            score += w
    sev = float(min(3.0, max(0.0, score / C.LEX_NORM)))
    return sev, sorted(set(hits)), float(score)


# --------------------------------------------------------------------------- #
# Signal B — resolution-time severity
# --------------------------------------------------------------------------- #
class RTScaler:
    """Fit percentile cut points once; reuse at inference (saved to disk)."""

    def __init__(self, cuts=None):
        self.cuts = cuts          # [p25, p50, p90]

    def fit(self, hours: pd.Series):
        vals = pd.to_numeric(hours, errors="coerce").dropna()
        if len(vals) >= 4:
            self.cuts = list(np.percentile(vals, C.RT_PERCENTILES))
        else:
            self.cuts = [1.0, 2.0, 3.0]   # degenerate fallback
        return self

    def transform(self, h):
        """Return (severity 0..3, known bool, percentile int)."""
        if h is None or (isinstance(h, float) and np.isnan(h)) or pd.isna(h):
            return 1.0, False, -1
        c25, c50, c90 = self.cuts
        if h <= c25:
            return 0.0, True, 12
        if h <= c50:
            return 1.0, True, 37
        if h <= c90:
            return 2.0, True, 70
        return 3.0, True, 95

    def to_dict(self):
        return {"cuts": self.cuts}

    @classmethod
    def from_dict(cls, d):
        return cls(cuts=d.get("cuts"))


# --------------------------------------------------------------------------- #
# Signal C — embedding urgency (lazy, optional)
# --------------------------------------------------------------------------- #
class EmbeddingScorer:
    """Wraps sentence-transformers. Degrades gracefully if unavailable."""

    def __init__(self):
        self.model = None
        self.high = None
        self.low = None
        self.available = False

    def load(self):
        try:
            from sentence_transformers import SentenceTransformer, util
            self._util = util
            self.model = SentenceTransformer(C.EMB_MODEL)
            self.high = self.model.encode(C.EMB_HIGH, convert_to_tensor=True)
            self.low = self.model.encode(C.EMB_LOW, convert_to_tensor=True)
            self.available = True
        except Exception as e:           # offline / not installed
            print(f"[signals] embedding signal disabled ({type(e).__name__}: {e})")
            self.available = False
        return self

    def score(self, text: str):
        """Return (severity 0..3, similarity float or None)."""
        if not self.available:
            return None, None
        v = self.model.encode(str(text), convert_to_tensor=True)
        hi = float(self._util.cos_sim(v, self.high).max())
        lo = float(self._util.cos_sim(v, self.low).max())
        diff = hi - lo                       # -1..1
        sev = float(min(3.0, max(0.0, (diff + 1) * 1.5)))
        return sev, round(diff, 3)


# --------------------------------------------------------------------------- #
# Fusion + mismatch derivation
# --------------------------------------------------------------------------- #
def _weights(use_emb: bool) -> dict:
    w = dict(C.FUSION_WEIGHTS)
    if not use_emb:
        w.pop("emb")
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}   # renormalize


def fuse(sev_lex, sev_rt, sev_emb, use_emb: bool):
    """Return the continuous fused severity score (un-rounded)."""
    w = _weights(use_emb)
    cont = w["lex"] * sev_lex + w["rt"] * sev_rt
    if use_emb:
        cont += w["emb"] * sev_emb
    return float(cont)


class SeverityCalibrator:
    """
    Maps the continuous fused score onto the 0..3 ordinal so that inferred
    severity is COMPARABLE to the assigned-priority scale. Cut points are the
    quantiles of the fused score at the cumulative proportions of the assigned
    priority distribution. Without this, a fused score that clusters near the
    middle would make every High/Critical ticket look like a False Alarm purely
    by construction. Fitted on training data and reused at inference.
    """

    def __init__(self, cuts=None):
        self.cuts = cuts          # [c1, c2, c3] continuous-score thresholds

    def fit(self, cont_scores, assigned_ords):
        cont = np.asarray(cont_scores, dtype=float)
        assigned = np.asarray(assigned_ords, dtype=int)
        n = max(len(assigned), 1)
        # cumulative proportion of tickets at or below each ordinal level
        cum = []
        running = 0
        for k in (0, 1, 2):
            running += int((assigned == k).sum())
            cum.append(running / n)
        # guard degenerate distributions
        cum = [min(max(p, 0.01), 0.99) for p in cum]
        self.cuts = [float(np.quantile(cont, p)) for p in cum]
        # enforce strictly increasing cuts
        for i in range(1, len(self.cuts)):
            if self.cuts[i] <= self.cuts[i - 1]:
                self.cuts[i] = self.cuts[i - 1] + 1e-6
        return self

    def transform(self, cont: float) -> int:
        c1, c2, c3 = self.cuts
        if cont <= c1:
            return 0
        if cont <= c2:
            return 1
        if cont <= c3:
            return 2
        return 3

    def to_dict(self):
        return {"cuts": self.cuts}

    @classmethod
    def from_dict(cls, d):
        return cls(cuts=d.get("cuts")) if d else None


def derive_mismatch(inferred_ord: int, assigned_ord: int):
    delta = inferred_ord - assigned_ord
    mismatch = int(abs(delta) >= C.MISMATCH_DELTA)
    mtype = ("Hidden Crisis" if delta > 0 else "False Alarm") if mismatch else "None"
    return mismatch, mtype, delta


# --------------------------------------------------------------------------- #
# Whole-frame pseudo-labeling (two-pass: signals -> calibrate -> mismatch)
# --------------------------------------------------------------------------- #
def generate_pseudo_labels(df: pd.DataFrame, rt_scaler: RTScaler = None,
                           embedder: EmbeddingScorer = None,
                           calibrator: "SeverityCalibrator" = None,
                           signals=("lex", "rt", "emb")):
    """
    Adds per-signal columns + calibrated inferred severity + mismatch label.
    Returns (df, rt_scaler, embedder, calibrator). Pass fitted objects at
    inference time so the mapping matches training.
    """
    df = df.copy()
    use_emb = "emb" in signals

    if rt_scaler is None:
        rt_scaler = RTScaler().fit(df["resolution_hours"])
    if use_emb and embedder is None:
        embedder = EmbeddingScorer().load()
    use_emb = use_emb and (embedder is not None and embedder.available)

    # pass 1 — per-signal severities + continuous fused score
    rows = []
    for _, r in df.iterrows():
        sl, kws, kw_w = s_lex(r["text"]) if "lex" in signals else (1.0, [], 0.0)
        sr, rk, rpct = rt_scaler.transform(r["resolution_hours"]) if "rt" in signals else (1.0, False, -1)
        se, sim = embedder.score(r["text"]) if use_emb else (None, None)
        cont = fuse(sl, sr, se if se is not None else 1.0, use_emb)
        rows.append(dict(
            sev_lex=round(sl, 3), kw=kws, kw_weight=round(kw_w, 2),
            sev_rt=round(sr, 3), rt_known=bool(rk), rt_pct=int(rpct),
            sev_emb=(round(se, 3) if se is not None else None), emb_sim=sim,
            inferred_cont=round(cont, 3), _cont=cont,
        ))

    # pass 2 — calibrate continuous -> ordinal, then mismatch
    if calibrator is None:
        calibrator = SeverityCalibrator().fit(
            [x["_cont"] for x in rows], df["assigned_ord"].tolist())
    assigned = df["assigned_ord"].tolist()
    for x, a in zip(rows, assigned):
        ordv = calibrator.transform(x.pop("_cont"))
        mm, mtype, delta = derive_mismatch(ordv, a)
        x.update(inferred_ord=int(ordv), inferred_sev=C.ORD_TO_SEVERITY[int(ordv)],
                 delta=int(delta), mismatch=int(mm), mismatch_type=mtype)

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out, rt_scaler, embedder, calibrator


# --------------------------------------------------------------------------- #
# Pseudo-label signal-agreement metric (required by spec §5)
# --------------------------------------------------------------------------- #
def signal_agreement(df: pd.DataFrame, a="sev_lex", b="sev_emb"):
    """Binarize two signals into low/elevated and report agreement % + kappa.
    Picks a threshold that actually splits each signal (median can be degenerate
    for sparse signals like the lexicon, where most values are 0)."""
    from sklearn.metrics import cohen_kappa_score
    sa, sb = df[a].dropna(), df[b].dropna()
    common = sa.index.intersection(sb.index)
    if len(common) < 2:
        return {"signal_a": a, "signal_b": b, "agreement": None, "kappa": None}

    def binarize(s):
        for thr in (s.median(), s.mean(), 0.0):
            bb = (s > thr).astype(int)
            if bb.nunique() == 2:
                return bb
        return (s > thr).astype(int)

    ab = binarize(df.loc[common, a])
    bb = binarize(df.loc[common, b])
    if ab.nunique() < 2 or bb.nunique() < 2:
        return {"signal_a": a, "signal_b": b,
                "agreement": round(float((ab == bb).mean()), 4), "kappa": None}
    return {
        "signal_a": a, "signal_b": b,
        "agreement": round(float((ab == bb).mean()), 4),
        "kappa": round(float(cohen_kappa_score(ab, bb)), 4),
    }
