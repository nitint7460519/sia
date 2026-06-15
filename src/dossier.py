"""
STAGE 3 — Evidence Dossier generation.

Every field is COPIED or COMPUTED from the ticket, never generated, so a
hallucination is structurally impossible. `verify_dossier` then proves it:
each evidence value must be traceable to a source field.
"""
from __future__ import annotations
from . import config as C


def build_dossier(row, confidence: float) -> dict:
    """
    `row` is a dict-like with the columns produced by
    signals.generate_pseudo_labels (plus the canonical ticket fields).
    """
    delta = int(row["delta"])
    mismatch_type = "Hidden Crisis" if delta > 0 else "False Alarm"

    evidence = []

    # keyword evidence — only emitted if terms ACTUALLY appeared in the text
    kws = list(row.get("kw") or [])
    if kws:
        evidence.append({
            "signal": "keyword",
            "value": ", ".join(kws),
            "weight": round(float(row.get("kw_weight", 0.0)), 2),
        })

    # resolution-time evidence — only if a real duration exists
    if bool(row.get("rt_known")):
        hrs = float(row["resolution_hours"])
        pct = int(row["rt_pct"])
        evidence.append({
            "signal": "resolution_time",
            "value": f"{hrs:.0f}h",
            "interpretation": f"{pct}th percentile -> "
                              f"{'elevated' if pct >= 50 else 'low'} severity",
        })

    # embedding evidence — only if the semantic signal ran
    if row.get("sev_emb") is not None and row.get("emb_sim") is not None:
        evidence.append({
            "signal": "embedding",
            "value": f"urgency_similarity={row['emb_sim']}",
            "interpretation": f"semantic match to {row['inferred_sev']} severity",
        })

    kw_txt = ", ".join(kws) if kws else "no escalation language"
    if bool(row.get("rt_known")):
        rt_txt = (f"resolution took {float(row['resolution_hours']):.0f}h "
                  f"({int(row['rt_pct'])}th pct)")
    else:
        rt_txt = "resolution time unavailable"
    analysis = (
        f"Ticket was assigned '{row['priority']}' priority, but its text shows "
        f"{kw_txt} and {rt_txt}, implying a '{row['inferred_sev']}' severity. "
        f"The {abs(delta)}-level gap indicates a {mismatch_type}."
    )

    return {
        "ticket_id": str(row["ticket_id"]),
        "assigned_priority": str(row["priority"]).capitalize(),
        "inferred_severity": str(row["inferred_sev"]),
        "mismatch_type": mismatch_type,
        "severity_delta": delta,
        "feature_evidence": evidence,
        "constraint_analysis": analysis,
        "confidence": round(float(confidence), 3),
    }


def verify_dossier(dossier: dict, row) -> list:
    """
    Returns a list of hallucination violations (empty == clean).
    Each evidence value must be traceable to a concrete source field.
    """
    violations = []
    text = str(row.get("text", "")).lower()
    for ev in dossier.get("feature_evidence", []):
        sig, val = ev.get("signal"), str(ev.get("value", ""))
        if sig == "keyword":
            for term in [t.strip().lower() for t in val.split(",") if t.strip()]:
                if term not in text:
                    violations.append(f"keyword '{term}' not present in ticket text")
        elif sig == "resolution_time":
            if not bool(row.get("rt_known")):
                violations.append("resolution_time evidence but no known duration")
        elif sig == "embedding":
            if row.get("sev_emb") is None:
                violations.append("embedding evidence but embedding signal did not run")
    # severity_delta must equal inferred_ord - assigned_ord
    if int(dossier["severity_delta"]) != int(row["inferred_ord"]) - int(row["assigned_ord"]):
        violations.append("severity_delta inconsistent with ordinals")
    return violations
