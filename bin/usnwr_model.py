"""
usnwr_model.py

A Python replica of the US News & World Report "Best National Universities"
2026 ranking methodology, built from the underlying formulas in
US_News_2026_Modeling_forTeams.xlsx.

Pipeline
--------
1. load_raw_data()          - load the 434-school input table
2. build_intermediate()     - derived fields (adjusted Pell rates, discounts, etc.)
3. build_ranking_factors()  - the 16 official USNWR "ranking factors"
4. build_zscores()          - population z-scores of each ranking factor
5. build_overall_scores()   - weighted sum -> rescaled 0-100 score -> rank

Weights (2026 methodology, sum to 1.00)
----------------------------------------
Six-Year Graduation Rate .......... 16.0%
First-Year Retention .............. 5.0%
Pell Grad Rate ..................... 5.5%
Pell Grad Rate Performance ......... 5.5%
Graduation Rate Performance ........10.0%
Peer Assessment ....................20.0%
Faculty Compensation ................6.0%
Pct Faculty Full-Time ...............2.0%
Student/Faculty Ratio ...............3.0%
SAT/ACT (25th-75th percentile) ......5.0%
Financial Resources per Student .....8.0%
Graduate Indebtedness ...............5.0%
Grads Earning > HS Grad .............5.0%
Citations per Publication ..........1.25%
Field-Weighted Citation Impact .....1.25%
Pubs in Top 5% Journals ............1.00%
Pubs in Top 25% Journals ...........0.50%
"""
from __future__ import annotations
import numpy as np
import logging
import time
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
TARGET_SCHOOL = "University of Virginia"  # school highlighted in run/scenario log messages

WEIGHTS = {
    "avg_six_yr_grad": 0.16,
    "retention": 0.05,
    "pell_grad_rate": 0.055,
    "pell_grad_perf": 0.055,
    "grad_rate_perf": 0.10,
    "peer_assessment": 0.20,
    "faculty_comp": 0.06,
    "pct_faculty_ft": 0.02,
    "student_faculty_ratio": 0.03,
    "satact": 0.05,
    "financial_resources": 0.08,
    "grad_indebtedness": 0.05,
    "grads_earning_more": 0.05,
    "citations_per_pub": 0.0125,
    "fwci": 0.0125,
    "pub_top5pct": 0.01,
    "pub_top25pct": 0.005,
}
# When SAT/ACT (satact) is unavailable for a school, USNWR drops that factor
# and folds its 5.0% weight into the six-year-graduation-rate factor (16% -> 21%).
GRAD_RATE_WEIGHT_NO_TESTS = 0.21

NUMERIC_COLS = [c for c in [
    "grad_avg4yr", "grad_2023", "grad_2024", "grad_2025", "grad_2026", "retention_avg",
    "pell_grad_2023", "pell_grad_2024", "pell_grad_2025", "pell_grad_2026",
    "nonpell_grad_2023", "nonpell_grad_2024", "nonpell_grad_2025", "nonpell_grad_2026",
    "pell_pct_2023", "pell_pct_2024", "pell_pct_2025", "pell_pct_2026",
    "pred_grad_2023", "pred_grad_2024", "pred_grad_2025", "pred_grad_2026",
    "peer_2025", "peer_2026",
    "sat_median", "sat25_prior", "sat75_prior", "sat25_current", "sat75_current",
    "sat_submit_2023", "sat_submit_prior", "sat_submit_current",
    "act_median", "act25_prior", "act75_prior", "act25_current", "act75_current",
    "act_submit_2023", "act_submit_prior", "act_submit_current",
    "grads_earning_more", "student_faculty_ratio", "pct_faculty_fulltime",
    "edu_expend_per_student", "citations_per_pub", "fwci", "pub_top5pct", "pub_top25pct",
    "total_papers_5yr", "avg_faculty_salary", "regional_price_parity", "avg_federal_indebtedness",
]]


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Builds the raw metric table either:
      - from the source tabs (Data, IPEDS 2023 Test Scores, Pell % 2015
        Cohort, Location) via data_lookup.build_raw_data() -- the default,
        and the faithful replica of the '2026' sheet's VLOOKUP formulas, or
      - from a pre-flattened CSV (path=..., e.g. a legacy raw_data.csv
        export), if you'd rather skip the source tabs entirely.
    """
    if path is None:
        from data_lookup import build_raw_data
        df = build_raw_data()
    else:
        df = pd.read_csv(path)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # 'Err' -> NaN
    return df


def _load_percentile_table(path: Path) -> pd.Series:
    t = pd.read_csv(path)
    return t.set_index("score")["percentile"]


SAT_PCTILE = _load_percentile_table(DATA_DIR / "sat_percentiles.csv")
ACT_PCTILE = _load_percentile_table(DATA_DIR / "act_percentiles.csv")


def _lookup_pctile(scores: pd.Series, table: pd.Series) -> pd.Series:
    """Excel VLOOKUP(score, table, 2, FALSE) equivalent - exact match only."""
    return scores.map(table.to_dict())


def _percentrank_inc(values: pd.Series) -> pd.Series:
    """Excel PERCENTRANK.INC across the full population, NaN-safe."""
    ranks = values.rank(method="average")
    n = values.notna().sum()
    if n <= 1:
        return pd.Series(np.nan, index=values.index)
    return (ranks - 1) / (n - 1)


def build_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates columns BF:CW of the '2026' sheet."""
    d = df.copy()

    def adj_pell(pell_col, pct_col):
        pell, pct = d[pell_col], d[pct_col]
        return np.where(pct < 50, pell + pct, pell + 0.5)

    d["adj_pell_2023"] = adj_pell("pell_grad_2023", "pell_pct_2023")
    d["adj_pell_2024"] = adj_pell("pell_grad_2024", "pell_pct_2024")
    d["adj_pell_2025"] = adj_pell("pell_grad_2025", "pell_pct_2025")
    d["adj_pell_2026"] = adj_pell("pell_grad_2026", "pell_pct_2026")
    d["avg_pell_grad"] = d[["adj_pell_2023", "adj_pell_2024", "adj_pell_2025", "adj_pell_2026"]].mean(axis=1)
    d["pell_grad_pctile"] = _percentrank_inc(d["avg_pell_grad"])

    for yr in ["2023", "2024", "2025", "2026"]:
        d[f"ratio_{yr}"] = d[f"pell_grad_{yr}"] / d[f"nonpell_grad_{yr}"]

    for yr in ["2023", "2024", "2025", "2026"]:
        col = f"ratio_{yr}"
        col_mean = d[col].mean(skipna=True)
        capped = d[col].where(d[col].notna(), col_mean)
        capped = np.where(capped > 1, 1, capped)
        d[f"ratio_{yr}_capped"] = capped

    for yr in ["2023", "2024", "2025", "2026"]:
        pct_col = f"pell_pct_{yr}"
        capped_col = f"ratio_{yr}_capped"
        d[f"ratio_{yr}_adj"] = np.where(
            d[pct_col] <= 50, d[capped_col] + d[pct_col] / 100, d[capped_col] + 0.5
        )

    d["avg_ratio"] = d[["ratio_2023_adj", "ratio_2024_adj", "ratio_2025_adj", "ratio_2026_adj"]].mean(axis=1)
    d["pell_perf_pctile"] = _percentrank_inc(d["avg_ratio"])

    d["financial_resources_pctile"] = _percentrank_inc(d["edu_expend_per_student"])
    d["grad_indebtedness_pctile"] = _percentrank_inc(d["avg_federal_indebtedness"])

    d["faculty_research_discount"] = np.where(
        d["total_papers_5yr"] < 5000, d["total_papers_5yr"] / 5000, 1.0
    )
    d["citations_per_pub_disc"] = d["faculty_research_discount"] * d["citations_per_pub"]
    d["fwci_disc"] = d["faculty_research_discount"] * d["fwci"]
    d["pub_top5pct_disc"] = d["faculty_research_discount"] * d["pub_top5pct"]
    d["pub_top25pct_disc"] = d["faculty_research_discount"] * d["pub_top25pct"]

    d["faculty_salary_adj"] = d["avg_faculty_salary"] / (d["regional_price_parity"] / 100)

    def round_to(x, base):
        return np.round(x / base) * base

    d["sat_median_prior"] = round_to((d["sat75_prior"] - d["sat25_prior"]) / 2 + d["sat25_prior"], 10)
    d["sat_median_current"] = round_to((d["sat75_current"] - d["sat25_current"]) / 2 + d["sat25_current"], 10)
    d["sat_pctile_prior"] = _lookup_pctile(d["sat_median_prior"], SAT_PCTILE)
    d["sat_pctile_current"] = _lookup_pctile(d["sat_median_current"], SAT_PCTILE)

    d["act_median_prior"] = round_to((d["act75_prior"] - d["act25_prior"]) / 2 + d["act25_prior"], 1)
    d["act_median_current"] = round_to((d["act75_current"] - d["act25_current"]) / 2 + d["act25_current"], 1)
    d["act_pctile_prior"] = _lookup_pctile(d["act_median_prior"], ACT_PCTILE)
    d["act_pctile_current"] = _lookup_pctile(d["act_median_current"], ACT_PCTILE)

    d["sat_weight_prior"] = d["sat_submit_prior"] / (d["sat_submit_prior"] + d["act_submit_prior"])
    d["sat_weight_current"] = d["sat_submit_current"] / (d["sat_submit_current"] + d["act_submit_current"])
    d["act_weight_prior"] = d["act_submit_prior"] / (d["sat_submit_prior"] + d["act_submit_prior"])
    d["act_weight_current"] = d["act_submit_current"] / (d["sat_submit_current"] + d["act_submit_current"])

    d["satact_sum_current"] = d["sat_submit_current"] + d["act_submit_current"]
    d["satact_bonus_current"] = d["satact_sum_current"] * 0.02
    d["satact_sum_prior"] = d["sat_submit_prior"] + d["act_submit_prior"]
    d["satact_bonus_prior"] = d["satact_sum_prior"] * 0.02

    current_weighted = (
        d["sat_pctile_current"] * d["sat_weight_current"]
        + d["act_pctile_current"] * d["act_weight_current"]
        + d["satact_bonus_current"]
    )
    prior_weighted = (
        d["sat_pctile_prior"] * d["sat_weight_prior"]
        + d["act_pctile_prior"] * d["act_weight_prior"]
        + d["satact_bonus_prior"]
    )
    d["satact_pctile_weighted"] = np.where(
        d["satact_sum_current"] >= 50, current_weighted,
        np.where(d["satact_sum_prior"] >= 50, prior_weighted, np.nan)
    )
    return d


def build_ranking_factors(d: pd.DataFrame) -> pd.DataFrame:
    """Replicates columns CX:DN - the 16 official ranking factors."""
    d = d.copy()
    d["avg_six_yr_grad"] = np.where(d["grad_avg4yr"] > 0, d["grad_avg4yr"], d["grad_avg4yr"].mean())
    d["retention"] = np.where(d["retention_avg"] > 0, d["retention_avg"], d["retention_avg"].mean())
    d["pell_grad_rate"] = d["pell_grad_pctile"].fillna(d["pell_grad_pctile"].mean())
    d["pell_grad_perf"] = d["pell_perf_pctile"].fillna(d["pell_perf_pctile"].mean())
    d["grad_rate_perf"] = (
        d["grad_2023"] / d["pred_grad_2023"] + d["grad_2024"] / d["pred_grad_2024"]
        + d["grad_2025"] / d["pred_grad_2025"] + d["grad_2026"] / d["pred_grad_2026"]
    ) / 4
    d["peer_assessment"] = d[["peer_2025", "peer_2026"]].mean(axis=1)
    faculty_comp_fallback = d["faculty_salary_adj"].mean() - d["faculty_salary_adj"].std(ddof=0)
    d["faculty_comp"] = np.where(d["faculty_salary_adj"] != 0, d["faculty_salary_adj"], faculty_comp_fallback)
    d["faculty_comp"] = d["faculty_comp"].fillna(faculty_comp_fallback)
    ft_fallback = d["pct_faculty_fulltime"].mean() - d["pct_faculty_fulltime"].std(ddof=0)
    d["pct_faculty_ft"] = d["pct_faculty_fulltime"].fillna(ft_fallback)
    d["student_faculty_ratio"] = d["student_faculty_ratio"]
    d["satact"] = d["satact_pctile_weighted"]
    d["financial_resources"] = d["financial_resources_pctile"]
    d["grad_indebtedness"] = d["grad_indebtedness_pctile"]
    d["grads_earning_more"] = np.minimum(d["grads_earning_more"], 95)
    d["citations_per_pub_f"] = d["citations_per_pub_disc"]
    d["fwci_f"] = d["fwci_disc"]
    d["pub_top5pct_f"] = d["pub_top5pct_disc"]
    d["pub_top25pct_f"] = d["pub_top25pct_disc"]
    return d


FACTOR_COLS = {
    "avg_six_yr_grad": "avg_six_yr_grad",
    "retention": "retention",
    "pell_grad_rate": "pell_grad_rate",
    "pell_grad_perf": "pell_grad_perf",
    "grad_rate_perf": "grad_rate_perf",
    "peer_assessment": "peer_assessment",
    "faculty_comp": "faculty_comp",
    "pct_faculty_ft": "pct_faculty_ft",
    "student_faculty_ratio": "student_faculty_ratio",
    "satact": "satact",
    "financial_resources": "financial_resources",
    "grad_indebtedness": "grad_indebtedness",
    "grads_earning_more": "grads_earning_more",
    "citations_per_pub": "citations_per_pub_f",
    "fwci": "fwci_f",
    "pub_top5pct": "pub_top5pct_f",
    "pub_top25pct": "pub_top25pct_f",
}
# factors where a *lower* raw value is better -> z-score sign is flipped
INVERTED_FACTORS = {"student_faculty_ratio", "grad_indebtedness"}


def build_zscores(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Population z-score (Excel STANDARDIZE with STDEV.P) for each factor."""
    d = d.copy()
    stats = {}
    for factor, col in FACTOR_COLS.items():
        mean = d[col].mean(skipna=True)
        std = d[col].std(ddof=0, skipna=True)
        stats[factor] = {"mean": mean, "std": std}
        z = (d[col] - mean) / std
        if factor in INVERTED_FACTORS:
            z = -z
        d[f"z_{factor}"] = z
    return d, stats


def build_overall_scores(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    z_cols = {f: f"z_{f}" for f in FACTOR_COLS}
    no_tests = d[z_cols["satact"]].isna()

    weighted_sum = pd.Series(0.0, index=d.index)
    for factor, wt in WEIGHTS.items():
        if factor == "satact":
            continue
        if factor == "avg_six_yr_grad":
            w = np.where(no_tests, GRAD_RATE_WEIGHT_NO_TESTS, wt)
        else:
            w = wt
        weighted_sum = weighted_sum + d[z_cols[factor]].fillna(np.nan) * w
    satact_term = np.where(no_tests, 0.0, d[z_cols["satact"]].fillna(0) * WEIGHTS["satact"])
    weighted_sum = weighted_sum + satact_term

    # If any *non-satact* factor is missing, the school can't be scored (mirrors
    # the Excel IFERROR->"Err" behavior for factors other than SAT/ACT).
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


def run_pipeline(raw: pd.DataFrame | None = None, target_school: str | None = TARGET_SCHOOL) -> pd.DataFrame:
    """
    Runs the full pipeline: raw data -> intermediate fields -> ranking
    factors -> z-scores -> weighted overall score -> rank.

    Logs when the run starts, when it finishes (with elapsed time), and the
    predicted score/rank for `target_school` (default: UVA), if it's present
    in the data. Pass target_school=None to skip that lookup/log line.
    """
    logger.info("Model run started")
    start = time.perf_counter()

    if raw is None:
        raw = load_raw_data()
    d = build_intermediate(raw)
    d = build_ranking_factors(d)
    d, _ = build_zscores(d)
    d = build_overall_scores(d)

    elapsed = time.perf_counter() - start
    logger.info("Model run finished in %.2fs (%d schools scored)",
                elapsed, d["overall_rank"].notna().sum())

    if target_school is not None:
        match = d[d["school"] == target_school]
        if not match.empty:
            row = match.iloc[0]
            logger.info(
                "%s predicted score: %.2f | predicted rank: %s",
                target_school, row["overall_score"],
                "n/a" if pd.isna(row["overall_rank"]) else f"{row['overall_rank']:.0f}",
            )
        else:
            logger.warning("%s not found in the scored data", target_school)

    return d


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


if __name__ == "__main__":
    log_file = Path(__file__).parent / "logs" / "usnwr_model_log.txt"
    _configure_logging(log_file)
    result = run_pipeline()
    uva = result[result["school"] == TARGET_SCHOOL].iloc[0]
    print(f"UVA overall score: {uva['overall_score']:.6f}")
    print(f"UVA overall rank:  {uva['overall_rank']:.0f}")
