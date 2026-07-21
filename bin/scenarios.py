"""
scenarios.py

Realistic, actionable "what-if" scenarios for UVA -- small, plausible moves in
raw underlying metrics (as opposed to the abstract 1-std-dev factor bumps in
sensitivity.py) -- to show what it actually takes to move the needle.

Run:  python scenarios.py
"""
from __future__ import annotations
import logging
import pandas as pd
from sensitivity import simulate_raw_change, SCHOOL

logger = logging.getLogger(__name__)

SCENARIOS = {
    "Peer assessment +0.1 (both survey years)": {"peer_2025": 0.1, "peer_2026": 0.1},
    "Peer assessment +0.2 (both survey years)": {"peer_2025": 0.2, "peer_2026": 0.2},
    "Six-year grad rate +1 pt (all 4 cohorts)": {
        "grad_2023": 1, "grad_2024": 1, "grad_2025": 1, "grad_2026": 1, "grad_avg4yr": 1
    },
    "Six-year grad rate +2 pt (all 4 cohorts)": {
        "grad_2023": 2, "grad_2024": 2, "grad_2025": 2, "grad_2026": 2, "grad_avg4yr": 2
    },
    "Student/faculty ratio -1 (14 -> 13)": {"student_faculty_ratio": -1},
    "Educational expenditure/student +$5,000": {"edu_expend_per_student": 5000},
    "Educational expenditure/student +$10,000": {"edu_expend_per_student": 10000},
    "Avg faculty salary +$10,000": {"avg_faculty_salary": 10000},
    "Pct faculty full-time +2 pt": {"pct_faculty_fulltime": 2},
    "Actual grad rate +2 pt (boosts grad-rate performance too)": {
        "grad_2023": 2, "grad_2024": 2, "grad_2025": 2, "grad_2026": 2
    },
    "Combo: peer +0.1, grad rate +1pt, S/F ratio -1": {
        "peer_2025": 0.1, "peer_2026": 0.1,
        "grad_2023": 1, "grad_2024": 1, "grad_2025": 1, "grad_2026": 1, "grad_avg4yr": 1,
        "student_faculty_ratio": -1,
    },
}


def run_all(scenarios: dict = SCENARIOS, school: str = SCHOOL) -> pd.DataFrame:
    rows = []
    for name, changes in scenarios.items():
        r = simulate_raw_change(changes, school=school, name=name)
        rows.append({
            "scenario": name,
            "before_score": r["before_score"],
            "after_score": r["after_score"],
            "score_delta": r["score_delta"],
            "before_rank": r["before_rank"],
            "after_rank": r["after_rank"],
            "rank_delta": r["rank_delta"],
        })
    return pd.DataFrame(rows).sort_values("rank_delta", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 55)
    df = run_all()
    print(df.to_string(index=False))
