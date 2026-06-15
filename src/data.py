"""
Data loading and preparation.

Turns whatever the raw CRM CSV looks like into a clean canonical frame with:
    ticket_id, subject, description, text, priority, assigned_ord,
    channel, ticket_type, product, resolution_hours, resolution_known
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

from . import config as C


def _norm(name: str) -> str:
    return re.sub(r"[\s_]+", " ", str(name)).strip().lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns to canonical names using fuzzy candidate matching."""
    lookup = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon, candidates in C.COLUMN_CANDIDATES.items():
        for cand in candidates:
            if _norm(cand) in lookup:
                out[canon] = df[lookup[_norm(cand)]]
                break
    return out


_PLACEHOLDER = re.compile(r"\{[^}]*\}")          # {product_purchased} etc.
_MULTISPACE = re.compile(r"\s+")
_GREETING = re.compile(r"^\s*(hi|hello|hey|dear)\s+support[,!.\s]*", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


def clean_text(s) -> str:
    if pd.isna(s):
        return ""
    s = _PLACEHOLDER.sub(" ", str(s))
    s = _GREETING.sub("", s)                     # drop "Hi Support," boilerplate
    s = _MULTISPACE.sub(" ", s)
    return s.strip()


def core_issue(desc) -> str:
    """First sentence of the (cleaned) description — the real issue, minus any
    trailing filler. Falls back to the whole cleaned string if no split."""
    c = clean_text(desc)
    if not c:
        return ""
    parts = _SENT_SPLIT.split(c)
    return parts[0].strip() if parts else c


def compute_resolution_hours(df: pd.DataFrame) -> pd.Series:
    """
    Derive resolution duration in hours.
    Priority: explicit numeric column -> timestamp difference -> NaN.
    """
    n = len(df)
    if "resolution_hours_raw" in df.columns:
        hrs = pd.to_numeric(df["resolution_hours_raw"], errors="coerce")
        if hrs.notna().any():
            return hrs

    if {"created_at", "resolved_at"}.issubset(df.columns):
        c = pd.to_datetime(df["created_at"], errors="coerce")
        r = pd.to_datetime(df["resolved_at"], errors="coerce")
        hrs = (r - c).dt.total_seconds() / 3600.0
        # negative/zero durations are data noise -> treat as unknown
        hrs = hrs.where(hrs > 0, np.nan)
        if hrs.notna().any():
            return hrs

    return pd.Series([np.nan] * n, index=df.index)


def _norm_priority(s) -> str:
    s = str(s).strip().lower()
    for p in C.PRIORITY_ORDER:
        if p in s:
            return p
    return "medium"        # safe default for malformed priority cells


def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full raw -> canonical clean frame."""
    df = standardize_columns(df_raw)

    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"T{i:06d}" for i in range(len(df))]
    for col in ("subject", "description", "channel", "ticket_type", "product"):
        if col not in df.columns:
            df[col] = ""
    if "priority" not in df.columns:
        raise ValueError(
            "No ticket-priority column found. SIA audits an assigned priority; "
            f"expected one of {C.COLUMN_CANDIDATES['priority']}."
        )

    df["subject"] = df["subject"].map(clean_text)
    # text = subject + the core issue sentence (drops greeting/trailing filler)
    core = df["description"].map(core_issue)
    df["description"] = df["description"].map(clean_text)
    df["text"] = (df["subject"] + ". " + core).str.strip(". ").str.strip()

    df["priority"] = df["priority"].map(_norm_priority)
    df["assigned_ord"] = df["priority"].map(C.PRIORITY_TO_ORD).astype(int)

    df["resolution_hours"] = compute_resolution_hours(df)
    df["resolution_known"] = df["resolution_hours"].notna()

    keep = ["ticket_id", "subject", "description", "text", "priority",
            "assigned_ord", "channel", "ticket_type", "product",
            "resolution_hours", "resolution_known"]
    return df[keep].reset_index(drop=True)


def load_prepared(path: str) -> pd.DataFrame:
    return prepare(pd.read_csv(path))
