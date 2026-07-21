"""
sensitivity.py

Scenario + sensitivity tooling built on top of usnwr_model.py

Two levels of analysis:

1. Factor-level "one-std-dev" sensitivity
   For each of the 16 official ranking factors, shift UVA's z-score by +1.0
   (i.e. move UVA from wherever it sits today to one population standard
   deviation better on that factor, holding every other factor fixed) and
   recompute the overall score. Because every factor is being tested in the
   same standardized (z-score) units, the resulting score deltas are directly
   comparable to each other -- this identifies the highest-*leverage* levers
   (weight matters, but so does how much room UVA has left on a 0-100ish
   scale for factors that are percentile-based, etc).

2. Raw-metric scenario simulation
   Lets you push a real, physical change into UVA's raw inputs (e.g.
   "+$5,000 educational expenditure per student", "-1.0 student/faculty
   ratio", "+0.1 peer assessment score") and re-runs the *entire* pipeline
   (because population means/medians/percentile ranks shift slightly too),
   returning the new overall score and rank.
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from usnwr_model import (
    load_raw_data, run_pipeline, WEIGHTS, FACTOR_COLS, INVERTED_FACTORS, TARGET_SCHOOL,
)

logger = logging.getLogger(__name__)

SCHOOL = TARGET_SCHOOL

FACTOR_LABELS = {
    "avg_six_yr_grad": "Six-Year Graduation Rate",
    "retention": "First-Year Retention Rate",
    "pell_grad_rate": "Pell Grad Rate (percentile)",
    "pell_grad_perf": "Pell vs Non-Pell Grad Gap (percentile)",
    "grad_rate_perf": "Graduation Rate Performance (actual/predicted)",
    "peer_assessment": "Peer Assessment Score",
    "faculty_comp": "Faculty Compensation (COL-adjusted)",
    "pct_faculty_ft": "% Faculty Full-Time",
    "student_faculty_ratio": "Student/Faculty Ratio",
    "satact": "SAT/ACT percentile",
    "financial_resources": "Financial Resources per Student (percentile)",
    "grad_indebtedness": "Graduate Indebtedness (percentile)",
    "grads_earning_more": "Grads Earning More Than HS Grad",
    "citations_per_pub": "Citations per Publication",
    "fwci": "Field-Weighted Citation Impact",
    "pub_top5pct": "Publications in Top 5% Journals",
    "pub_top25pct": "Publications in Top 25% Journals",
}


def one_std_dev_sensitivity(school: str = SCHOOL) -> pd.DataFrame:
    """
    For every ranking factor, ask: "if UVA's z-score on this factor were
    exactly +1.0 higher (a one-population-std-dev improvement), all else
    equal, what would UVA's overall score/rank become?"
    """
    base = run_pipeline()
    base_row = base[base.school == school].iloc[0]
    base_score = base_row["overall_score"]
    base_rank = base_row["overall_rank"]

    records = []
    for factor in FACTOR_COLS:
        d = run_pipeline(load_raw_data(), target_school=None)  # internal recompute; base run above already logged
        idx = d.index[d.school == school][0]
        z_col = f"z_{factor}"
        d.loc[idx, z_col] = d.loc[idx, z_col] + 1.0

        # Recompute just the aggregation step from the bumped z-scores
        from usnwr_model import build_overall_scores
        # build_overall_scores expects the full frame with z_ columns already set;
        # re-run only the final aggregation using the already-computed factor frame.
        d2 = _rebuild_overall_from_frame(d)
        new_row = d2.loc[idx]

        raw_col = FACTOR_COLS[factor]
        std = base[raw_col].std(ddof=0, skipna=True)
        sign = -1 if factor in INVERTED_FACTORS else 1
        real_world_move = std * sign

        records.append({
            "factor": FACTOR_LABELS[factor],
            "weight": WEIGHTS[factor],
            "current_z": base_row[z_col],
            "1_std_dev_raw_units": real_world_move,
            "new_score": new_row["overall_score"],
            "score_delta": new_row["overall_score"] - base_score,
            "new_rank": new_row["overall_rank"],
            "rank_delta": base_rank - new_row["overall_rank"],  # positive = improvement
        })

    out = pd.DataFrame(records).sort_values("score_delta", ascending=False).reset_index(drop=True)
    out["current_score"] = base_score
    out["current_rank"] = base_rank
    return out


def _rebuild_overall_from_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Re-run only the weighted-sum -> rescale -> rank steps on a frame that
    already has z_ columns (some possibly hand-edited)."""
    d = d.copy()
    z_cols = {f: f"z_{f}" for f in FACTOR_COLS}
    no_tests = d[z_cols["satact"]].isna()
    from usnwr_model import GRAD_RATE_WEIGHT_NO_TESTS
    weighted_sum = pd.Series(0.0, index=d.index)
    for factor, wt in WEIGHTS.items():
        if factor == "satact":
            continue
        w = np.where(no_tests, GRAD_RATE_WEIGHT_NO_TESTS, wt) if factor == "avg_six_yr_grad" else wt
        weighted_sum = weighted_sum + d[z_cols[factor]] * w
    satact_term = np.where(no_tests, 0.0, d[z_cols["satact"]].fillna(0) * WEIGHTS["satact"])
    weighted_sum = weighted_sum + satact_term

    required = [z_cols[f] for f in FACTOR_COLS if f != "satact"]
    any_missing_required = d[required].isna().any(axis=1)
    weighted_sum = weighted_sum.where(~any_missing_required, np.nan)

    d["overall_sum_weighted_z"] = weighted_sum
    offset = abs(d["overall_sum_weighted_z"].min(skipna=True))
    d["rescaled_sum"] = d["overall_sum_weighted_z"] + offset
    max_rescaled = d["rescaled_sum"].max(skipna=True)
    d["overall_score"] = d["rescaled_sum"] / max_rescaled * 100
    d["overall_score_rounded"] = d["overall_score"].round(0)
    d["overall_rank"] = d["overall_score_rounded"].rank(method="min", ascending=False)
    return d


def simulate_raw_change(changes: dict, school: str = SCHOOL, name: str | None = None) -> dict:
    """
    Apply real-world changes to UVA's raw input columns (see usnwr_model.NUMERIC_COLS
    for valid keys) and re-run the FULL pipeline (population stats shift too).
    Returns a before/after comparison dict.

    `name` is an optional human-readable scenario label used only in the log
    messages (e.g. "Peer assessment +0.1"); if omitted, the raw `changes`
    dict is logged instead.
    """
    label = name if name is not None else str(changes)
    logger.info("Scenario run started: %s (%s)", label, school)

    raw = load_raw_data()
    idx = raw.index[raw.school == school][0]
    before = run_pipeline(raw.copy(), target_school=None)
    before_row = before[before.school == school].iloc[0]

    raw2 = raw.copy()
    for col, delta in changes.items():
        raw2.loc[idx, col] = raw2.loc[idx, col] + delta

    after = run_pipeline(raw2, target_school=None)
    after_row = after[after.school == school].iloc[0]

    result = {
        "changes": changes,
        "before_score": before_row["overall_score"],
        "before_rank": before_row["overall_rank"],
        "after_score": after_row["overall_score"],
        "after_rank": after_row["overall_rank"],
        "score_delta": after_row["overall_score"] - before_row["overall_score"],
        "rank_delta": before_row["overall_rank"] - after_row["overall_rank"],
    }

    logger.info(
        "Scenario run finished: %s -- score %.2f -> %.2f (%+.2f) | rank %.0f -> %.0f (%+.0f)",
        label, result["before_score"], result["after_score"], result["score_delta"],
        result["before_rank"], result["after_rank"], result["rank_delta"],
    )
    return result


def rank_needed_gap(target_rank: int, school: str = SCHOOL) -> dict:
    """How many weighted-z-score points does UVA need to reach a target rank?"""
    d = run_pipeline()
    d_scored = d.dropna(subset=["overall_rank"]).sort_values("overall_rank")
    target_row = d_scored[d_scored["overall_rank"] <= target_rank].iloc[-1]
    uva_row = d[d.school == school].iloc[0]
    return {
        "target_rank": target_rank,
        "target_school_at_cutoff": target_row["school"],
        "target_score": target_row["overall_score"],
        "uva_score": uva_row["overall_score"],
        "score_gap": target_row["overall_score"] - uva_row["overall_score"],
        "weighted_z_gap": target_row["overall_sum_weighted_z"] - uva_row["overall_sum_weighted_z"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 45)

    print("=" * 100)
    print("ONE-STD-DEV SENSITIVITY  (each row: UVA improves ONE factor by +1 population std dev)")
    print("=" * 100)
    sens = one_std_dev_sensitivity()
    print(sens[["factor", "weight", "current_z", "1_std_dev_raw_units",
                "score_delta", "rank_delta"]].to_string(index=False))

    print()
    print("=" * 100)
    print("GAP TO TOP 20")
    print("=" * 100)
    print(rank_needed_gap(20))
