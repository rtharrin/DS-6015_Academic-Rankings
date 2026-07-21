"""
run_pipeline.py

Top-level orchestration script. Runs, in order:
  1. The core model (usnwr_model.run_pipeline) -> every school's score & rank
  2. The one-std-dev factor sensitivity analysis (sensitivity.one_std_dev_sensitivity)
  3. The realistic scenario simulations (scenarios.run_all)

Logs the start/finish of each stage (and of the run as a whole); the model
and scenario stages additionally log UVA's predicted score/rank on every
underlying model run, via the logging already built into usnwr_model.py and
sensitivity.py.

Results are saved to CSV in this directory so they can be inspected or
loaded elsewhere without re-running anything:
  full_rankings.csv, sensitivity_one_std_dev.csv, scenario_results.csv

Run:  python run_pipeline.py
"""
from __future__ import annotations
import logging
import time
from pathlib import Path

from usnwr_model import run_pipeline, TARGET_SCHOOL
from sensitivity import one_std_dev_sensitivity, rank_needed_gap
from scenarios import run_all, SCENARIOS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Model Outputs"


def main(target_school: str = TARGET_SCHOOL, top_n_gap: int = 20) -> dict:
    logger.info("=" * 70)
    logger.info("Pipeline run started")
    run_start = time.perf_counter()

    # 1. Core model: score & rank every school -----------------------------
    logger.info("Stage 1/3: scoring all schools")
    rankings = run_pipeline(target_school=target_school)
    rankings_path = OUTPUT_DIR / "full_rankings.csv"
    rankings.sort_values("overall_rank").to_csv(rankings_path, index=False)
    logger.info("Stage 1/3 complete -> %s", rankings_path.name)

    # 2. One-std-dev factor sensitivity -------------------------------------
    logger.info("Stage 2/3: running one-std-dev factor sensitivity for %s", target_school)
    sensitivity = one_std_dev_sensitivity(school=target_school)
    sensitivity_path = OUTPUT_DIR / "sensitivity_one_std_dev.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    top_lever = sensitivity.iloc[0]
    logger.info(
        "Stage 2/3 complete -> %s (top lever: %s, +%.2f score / +%.0f rank)",
        sensitivity_path.name, top_lever["factor"], top_lever["score_delta"], top_lever["rank_delta"],
    )

    # 3. Realistic actionable scenarios --------------------------------------
    logger.info("Stage 3/3: running %d scenarios for %s", len(SCENARIOS), target_school)
    scenario_results = run_all(scenarios=SCENARIOS, school=target_school)
    scenario_path = OUTPUT_DIR / "scenario_results.csv"
    scenario_results.to_csv(scenario_path, index=False)
    best_scenario = scenario_results.iloc[0]
    logger.info(
        "Stage 3/3 complete -> %s (best scenario: %s, rank %.0f -> %.0f)",
        scenario_path.name, best_scenario["scenario"], best_scenario["before_rank"], best_scenario["after_rank"],
    )

    gap = rank_needed_gap(top_n_gap, school=target_school)

    elapsed = time.perf_counter() - run_start
    uva_row = rankings[rankings["school"] == target_school].iloc[0]
    logger.info(
        "Pipeline run finished in %.2fs | %s predicted score: %.2f | predicted rank: %.0f",
        elapsed, target_school, uva_row["overall_score"], uva_row["overall_rank"],
    )
    logger.info("=" * 70)

    return {
        "rankings": rankings,
        "sensitivity": sensitivity,
        "scenario_results": scenario_results,
        "gap_to_top_n": gap,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = main()

    print()