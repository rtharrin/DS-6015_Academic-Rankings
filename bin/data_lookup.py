"""
data_lookup.py

Builds usnwr_model.py's raw metric table directly from the 
source tabs; Data, IPEDS 2023 Test Scores, Pell % 2015 Cohort, and
Location 

Source tabs:

  school_roster.csv
  data_long.csv
  ipeds_2023_test_scores.csv
  pell_2015_cohort.csv
  location.csv 

Exports raw_data.csv, which is the Python equivalent of the '2026' sheet's VLOOKUP formulas.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"

# ---------------------------------------------------------------------------
# Field specification 
# ---------------------------------------------------------------------------
# type "data": pulls from data_long.csv using (metric_description, year)
# type "test_scores": pulls from ipeds_2023_test_scores.csv using `field`
# type "pell_2015": pulls from pell_2015_cohort.csv, multiplied by 100
# type "location": pulls from location.csv, default 100 if missing
# `missing` says what Excel's IFERROR/blank fallback resolves to: "err" -> NaN,
# "blank" -> NaN, or a literal number (e.g. 0 for the citation metrics).
FIELD_SPEC = [
    {"out": "grad_2023", "type": "data", "metric": "6-year graduation rate (single cohort)", "year": "y2023", "missing": "err"},
    {"out": "grad_2024", "type": "data", "metric": "6-year graduation rate (single cohort)", "year": "y2024", "missing": "err"},
    {"out": "grad_2025", "type": "data", "metric": "6-year graduation rate (single cohort)", "year": "y2025", "missing": "err"},
    {"out": "grad_2026", "type": "data", "metric": "6-year graduation rate (single cohort)", "year": "y2026", "missing": "err"},
    {"out": "retention_avg", "type": "data", "metric": "Average first year student retention rate", "year": "y2026", "missing": "err"},

    {"out": "pell_grad_2023", "type": "data", "metric": "6-Year Graduation Rate Of Students Who Received A Pell Grant", "year": "y2023", "missing": "err"},
    {"out": "pell_grad_2024", "type": "data", "metric": "6-Year Graduation Rate Of Students Who Received A Pell Grant", "year": "y2024", "missing": "err"},
    {"out": "pell_grad_2025", "type": "data", "metric": "6-Year Graduation Rate Of Students Who Received A Pell Grant", "year": "y2025", "missing": "err"},
    {"out": "pell_grad_2026", "type": "data", "metric": "6-Year Graduation Rate Of Students Who Received A Pell Grant", "year": "y2026", "missing": "err"},

    {"out": "nonpell_grad_2023", "type": "data", "metric": "6-year graduation rate of students who did not receive a Pell Grant ", "year": "y2023", "missing": "err"},
    {"out": "nonpell_grad_2024", "type": "data", "metric": "6-year graduation rate of students who did not receive a Pell Grant ", "year": "y2024", "missing": "err"},
    {"out": "nonpell_grad_2025", "type": "data", "metric": "6-year graduation rate of students who did not receive a Pell Grant ", "year": "y2025", "missing": "err"},
    {"out": "nonpell_grad_2026", "type": "data", "metric": "6-year graduation rate of students who did not receive a Pell Grant ", "year": "y2026", "missing": "err"},

    {"out": "pell_pct_2023", "type": "pell_2015", "missing": "err"},  # special: own sheet, not Data tab
    {"out": "pell_pct_2024", "type": "data", "metric": "Pell % of Cohort", "year": "y2024", "missing": "err"},
    {"out": "pell_pct_2025", "type": "data", "metric": "Pell % of Cohort", "year": "y2025", "missing": "err"},
    {"out": "pell_pct_2026", "type": "data", "metric": "Pell % of Cohort", "year": "y2026", "missing": "err"},

    {"out": "pred_grad_2023", "type": "data", "metric": "Predicted Graduation Rate", "year": "y2023", "missing": "err"},
    {"out": "pred_grad_2024", "type": "data", "metric": "Predicted Graduation Rate", "year": "y2024", "missing": "err"},
    {"out": "pred_grad_2025", "type": "data", "metric": "Predicted Graduation Rate", "year": "y2025", "missing": "err"},
    {"out": "pred_grad_2026", "type": "data", "metric": "Predicted Graduation Rate", "year": "y2026", "missing": "err"},

    {"out": "peer_2025", "type": "data", "metric": "Peer assessment score", "year": "y2025", "missing": "err"},
    {"out": "peer_2026", "type": "data", "metric": "Peer assessment score", "year": "y2026", "missing": "err"},

    {"out": "sat_median", "type": "test_scores", "field": "SAT Composite 50th Percentile", "missing": "blank"},
    {"out": "sat25_prior", "type": "data", "metric": "SAT 25th Percentile", "year": "y2025", "missing": "blank"},
    {"out": "sat75_prior", "type": "data", "metric": "SAT 75th Percentile", "year": "y2025", "missing": "blank"},
    {"out": "sat25_current", "type": "data", "metric": "SAT 25th Percentile", "year": "y2026", "missing": "blank"},
    {"out": "sat75_current", "type": "data", "metric": "SAT 75th Percentile", "year": "y2026", "missing": "blank"},

    {"out": "sat_submit_2023", "type": "test_scores",
     "field": "ADM2023.Percent of first-time degree/certificate-seeking students submitting SAT scores", "missing": "blank"},
    {"out": "sat_submit_prior", "type": "data", "metric": "% new entrants students submitting SAT scores", "year": "y2025", "missing": "blank"},
    {"out": "sat_submit_current", "type": "data", "metric": "% new entrants students submitting SAT scores", "year": "y2026", "missing": "blank"},

    {"out": "act_median", "type": "test_scores", "field": "ADM2023.ACT Composite 50th percentile score", "missing": "blank"},
    {"out": "act25_prior", "type": "data", "metric": "ACT 25th Percentile", "year": "y2025", "missing": "blank"},
    {"out": "act75_prior", "type": "data", "metric": "ACT 75th Percentile", "year": "y2025", "missing": "blank"},
    {"out": "act25_current", "type": "data", "metric": "ACT 25th Percentile", "year": "y2026", "missing": "blank"},
    {"out": "act75_current", "type": "data", "metric": "ACT 75th Percentile", "year": "y2026", "missing": "blank"},

    {"out": "act_submit_2023", "type": "test_scores",
     "field": "ADM2023.Percent of first-time degree/certificate-seeking students submitting ACT scores", "missing": "blank"},
    {"out": "act_submit_prior", "type": "data", "metric": "% new entrants students submitting ACT scores", "year": "y2025", "missing": "blank"},
    {"out": "act_submit_current", "type": "data", "metric": "% new entrants students submitting ACT scores", "year": "y2026", "missing": "blank"},

    {"out": "grads_earning_more", "type": "data", "metric": "College grads earning more than a HS grad", "year": "y2026", "missing": "blank"},
    {"out": "student_faculty_ratio", "type": "data", "metric": "Student/faculty ratio", "year": "y2026", "missing": "blank"},
    {"out": "pct_faculty_fulltime", "type": "data", "metric": "faculty who are full-time", "year": "y2026", "missing": "blank"},
    {"out": "edu_expend_per_student", "type": "data", "metric": "Educational expenditures per student", "year": "y2026", "missing": "blank"},

    {"out": "citations_per_pub", "type": "data", "metric": "Citations per publication+", "year": "y2026", "missing": 0},
    {"out": "fwci", "type": "data", "metric": "Field-Weighted Citation Impact+", "year": "y2026", "missing": 0},
    {"out": "pub_top5pct", "type": "data", "metric": "Publication share in the Top 5% of Journals by CiteScore+", "year": "y2026", "missing": 0},
    {"out": "pub_top25pct", "type": "data", "metric": "Publication share in the Top 25% of Journals by CiteScore+", "year": "y2026", "missing": 0},
    {"out": "total_papers_5yr", "type": "data", "metric": "Total Papers Published (5-year period)", "year": "y2026", "missing": "blank"},
    {"out": "avg_faculty_salary", "type": "data", "metric": "Average Faculty Salary", "year": "y2026", "missing": "blank"},

    {"out": "regional_price_parity", "type": "location", "missing": 100},
    {"out": "avg_federal_indebtedness", "type": "data", "metric": "Average Federal Indebtedness of graduating class", "year": "y2026", "missing": "blank"},
]


def _resolve_missing(series: pd.Series, missing) -> pd.Series:
    if missing in ("err", "blank"):
        return series  # already NaN where absent
    return series.fillna(missing)


def load_sources(data_dir: Path = DATA_DIR) -> dict:
    roster = pd.read_csv(data_dir / "school_roster.csv")
    data_long = pd.read_csv(data_dir / "data_long.csv")
    test_scores = pd.read_csv(data_dir / "ipeds_2023_test_scores.csv")
    pell_2015 = pd.read_csv(data_dir / "pell_2015_cohort.csv")
    location = pd.read_csv(data_dir / "location.csv")
    return {
        "roster": roster, "data_long": data_long, "test_scores": test_scores,
        "pell_2015": pell_2015, "location": location,
    }


def _vlookup_data(data_long: pd.DataFrame, ipeds_ids: pd.Series, metric: str, year: str) -> pd.Series:
    """Replicates VLOOKUP($C&hdr8, Data!$C:$col, n, FALSE): exact match on
    (ipeds_id, metric_description), returning the requested year's column.

    Excel's VLOOKUP with FALSE (exact match) is case-INsensitive, and a few
    metric-description strings on the '2026' sheet differ from the 'Data'
    tab only in casing (e.g. sheet says "Predicted Graduation Rate", Data
    tab says "Predicted graduation rate") -- so we match on lowercased text,
    same as Excel would.
    """
    mask = data_long["metric_description"].str.lower() == metric.lower()
    sub = data_long.loc[mask, ["ipeds_id", year]]
    sub = sub.drop_duplicates(subset="ipeds_id", keep="first").set_index("ipeds_id")[year]
    return ipeds_ids.map(sub)


def build_raw_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Recreates the raw_data.csv table by resolving every field in
    FIELD_SPEC against the actual source tabs -- the Python equivalent of
    the '2026' sheet's VLOOKUP formulas."""
    src = load_sources(data_dir)
    roster = src["roster"]
    ids = roster["ipeds_id"]

    out = pd.DataFrame({
        "rank_year": roster["rank_year"],
        "ipeds_id": ids,
        "school": roster["school"],
    })

    test_scores_idx = src["test_scores"].set_index("unitid")
    pell_2015 = src["pell_2015"].copy()
    pell_2015["pell_pct_of_cohort"] = pd.to_numeric(pell_2015["pell_pct_of_cohort"], errors="coerce")  # '#DIV/0!' -> NaN
    pell_idx = pell_2015.drop_duplicates(subset="unitid", keep="first").set_index("unitid")["pell_pct_of_cohort"]
    location_idx = src["location"].drop_duplicates(subset="unitid", keep="first").set_index("unitid")["rpp"]

    for spec in FIELD_SPEC:
        if spec["type"] == "data":
            col = _vlookup_data(src["data_long"], ids, spec["metric"], spec["year"])
        elif spec["type"] == "test_scores":
            col = ids.map(test_scores_idx[spec["field"]])
        elif spec["type"] == "pell_2015":
            col = ids.map(pell_idx) * 100
        elif spec["type"] == "location":
            col = ids.map(location_idx)
        else:
            raise ValueError(f"unknown field type: {spec['type']}")
        out[spec["out"]] = _resolve_missing(pd.to_numeric(col, errors="coerce"), spec["missing"])

    # Derived (formula, not a lookup): 4-year average of single-cohort grad rates
    out["grad_avg4yr"] = out[["grad_2023", "grad_2024", "grad_2025", "grad_2026"]].mean(axis=1)

    return out


if __name__ == "__main__":
    df = build_raw_data()
    df.to_csv(DATA_DIR / "raw_data.csv", index=False)
    print(df.shape)
    uva = df[df.school == "University of Virginia"].iloc[0]
    print(uva[["grad_avg4yr", "retention_avg", "peer_2025", "peer_2026",
               "student_faculty_ratio", "avg_faculty_salary", "regional_price_parity",
               "sat_median", "act_median"]])

