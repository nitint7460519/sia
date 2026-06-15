"""
Central configuration for the Support Integrity Auditor (SIA).

Everything tunable lives here so the rest of the code stays clean and the
README ablation can be reproduced by flipping a few values.
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Severity ordinal scale  (single source of truth)
# --------------------------------------------------------------------------- #
PRIORITY_ORDER = ["low", "medium", "high", "critical"]
PRIORITY_TO_ORD = {p: i for i, p in enumerate(PRIORITY_ORDER)}
ORD_TO_SEVERITY = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

# --------------------------------------------------------------------------- #
# Robust column mapping: canonical_name -> list of accepted raw column names
# (matched case-insensitively, spaces/underscores ignored). This is what makes
# the loader survive the real Kaggle schema vs. the spec's logical names.
# --------------------------------------------------------------------------- #
COLUMN_CANDIDATES = {
    "ticket_id":   ["ticket id", "ticketid", "id"],
    "subject":     ["ticket subject", "subject"],
    "description": ["ticket description", "description", "body", "text"],
    "priority":    ["ticket priority", "priority level", "priority"],
    "channel":     ["ticket channel", "channel", "source"],
    "ticket_type": ["ticket type", "issue category", "type", "category"],
    "product":     ["product purchased", "product", "product_purchased"],
    "email":       ["customer email", "email"],
    # timestamps used to derive resolution time
    "created_at":  ["date of purchase", "submission date", "created",
                    "created at", "open date", "first response time"],
    "resolved_at": ["time to resolution", "resolved", "resolved at",
                    "close date", "closed at"],
    # if the dataset already ships a numeric duration, we use it directly
    "resolution_hours_raw": ["resolution time hours", "resolution time",
                             "resolution hours", "resolution_time",
                             "time_to_resolution_hours"],
}

# --------------------------------------------------------------------------- #
# STAGE 1 — pseudo-label signals
# --------------------------------------------------------------------------- #
# Lexicon: term -> weight. Negation within a small left window cancels
# positive-weight hits (see signals.s_lex).
LEXICON = {
    # critical (weight 3)
    "urgent": 3, "asap": 3, "immediately": 3, "critical": 3, "emergency": 3,
    "outage": 3, "down": 3, "cannot access": 3, "can't access": 3,
    "data loss": 3, "breach": 3, "hacked": 3, "crash": 3, "crashed": 3,
    "not working": 3, "doesn't work": 3, "broken": 3, "escalate": 3,
    "legal": 3, "lawsuit": 3, "fraud": 3, "unauthorized": 3, "lost money": 3,
    "production down": 3, "system down": 3, "all users": 3,
    # moderate (weight 1.5)
    "slow": 1.5, "delay": 1.5, "delayed": 1.5, "error": 1.5, "fail": 1.5,
    "failed": 1.5, "issue": 1.5, "problem": 1.5, "stuck": 1.5, "bug": 1.5,
    "glitch": 1.5, "unable": 1.5, "wrong": 1.5,
    # trivial (negative weight)
    "question": -1, "how to": -1, "how do i": -1, "inquiry": -1,
    "feedback": -1, "thank": -1, "thanks": -1, "suggestion": -1,
    "just wondering": -1, "no rush": -1, "whenever": -1,
}
NEGATORS = {"no", "not", "without", "isnt", "isn't", "never", "cannot be",
            "no longer", "nothing", "wasnt", "wasn't", "dont", "don't"}
LEX_NORM = 3.0          # divide raw weighted score by this, then clamp to [0,3]
LEX_NEG_WINDOW = 2      # tokens to the left checked for negation

# Embedding prototypes (semantic, keyword-independent -> adversarial defense)
EMB_HIGH = [
    "the system is completely down and we are losing money",
    "urgent data breach, customer information exposed",
    "complete outage, nothing works for any user",
    "critical failure blocking the entire production environment",
    "payment failed and I was charged twice, this is an emergency",
]
EMB_LOW = [
    "just a quick question about how something works",
    "minor cosmetic feedback, no rush at all",
    "how do I change a setting in my account",
    "thanks for the help, everything is fine now",
    "a small suggestion for a future feature",
]
EMB_MODEL = "all-MiniLM-L6-v2"

# Resolution-time percentiles (severity proxy: longer == harder/more severe)
RT_PERCENTILES = [25, 50, 90]   # -> buckets 0,1,2,3

# Fusion weights (must sum to 1.0). Reweighted automatically if a signal is
# unavailable at runtime (e.g. embedding model could not load).
FUSION_WEIGHTS = {"lex": 0.40, "rt": 0.25, "emb": 0.35}

# Mismatch decision
MISMATCH_DELTA = 2      # |inferred - assigned| >= this  ->  mismatch

# --------------------------------------------------------------------------- #
# STAGE 2 — classifier
# --------------------------------------------------------------------------- #
MODEL_BACKEND = "deberta"          # "deberta" (spec-compliant) | "baseline"
HF_MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN = 256
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 16
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Metadata features serialized into the text (text + structured -> required)
META_FEATURES = ["channel", "ticket_type", "resolution_hours"]

# --------------------------------------------------------------------------- #
# Verification thresholds (from the spec)
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "accuracy": 0.83,
    "macro_f1": 0.82,
    "per_class_recall": 0.78,
}
