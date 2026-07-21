# USNWR "Best National Universities" 2026 — Python Methodology Model

A full Python re-implementation of the ranking methodology encoded in
`US_News_2026_Modeling_forTeams.xlsx`, plus sensitivity tooling to answer
"what would it actually take to move UVA's rank?"

## Files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Orchestration script: runs the model, the one-std-dev sensitivity analysis, and the scenario simulations in one go, with logging at every stage, and writes `full_rankings.csv`, `sensitivity_one_std_dev.csv`, and `scenario_results.csv` |
| `data_lookup.py` | Rebuilds the raw metric table directly from the source tabs, replicating every VLOOKUP formula on the `2026` sheet  |
| `usnwr_model.py` | Core pipeline: raw data -> intermediate fields -> 17 official ranking factors -> z-scores -> weighted overall score -> rank |
| `sensitivity.py` | Factor-level "+1 std dev" sensitivity analysis (apples-to-apples lever comparison) and a `simulate_raw_change()` scenario engine |
| `scenarios.py` | A set of realistic, small, actionable "what-if" moves for UVA (peer score +0.1, grad rate +1pt, etc.) |
| `school_roster.csv` | The 434-school index (`rank_year`, `ipeds_id`, `published_rank`, `school`) |
| `data_long.csv` | The `Data` tab, long format: one row per (school, metric), with `y2023..y2026` value columns — the table ~40 of the ~50 raw metrics are looked up from |
| `ipeds_2023_test_scores.csv` | The `IPEDS 2023 Test Scores` tab verbatim — SAT/ACT medians and submission rates, keyed by `unitid` |
| `pell_2015_cohort.csv` | The `Pell % 2015 Cohort` tab (unitid, Pell % of cohort) |
| `location.csv` | The `Location` tab (unitid, Regional Price Parity) |
| `sat_percentiles.csv` / `act_percentiles.csv` | SAT/ACT score-to-percentile lookup tables (from the `Test Score Percentiles` sheet) |



## Quick start

```bash
pip install pandas numpy
python run_pipeline.py     # runs everything: model + sensitivity + scenarios, with logging
python usnwr_model.py      # prints UVA's overall score & rank
python sensitivity.py      # 1-std-dev factor leverage ranking + gap-to-top-20
python scenarios.py        # realistic scenario simulations
```

```python
from usnwr_model import run_pipeline
df = run_pipeline()                 # full 434-school scored table
df.sort_values("overall_rank").head(25)[["school", "overall_score", "overall_rank"]]
```

```python
from sensitivity import simulate_raw_change
simulate_raw_change({"peer_2025": 0.1, "peer_2026": 0.1})   # what if UVA's peer score rose 0.1?
```

## Methodology (17 factors, weights sum to 100%)

| Factor | Weight |
|---|---|
| Peer Assessment | 20.0% |
| Six-Year Graduation Rate | 16.0% |
| Graduation Rate Performance (actual vs predicted) | 10.0% |
| Financial Resources per Student | 8.0% |
| Faculty Compensation | 6.0% |
| Pell Grad Rate | 5.5% |
| Pell Grad Rate Performance | 5.5% |
| First-Year Retention | 5.0% |
| SAT/ACT Scores | 5.0% |
| Graduate Indebtedness | 5.0% |
| Grads Earning More Than a HS Grad | 5.0% |
| Student/Faculty Ratio | 3.0% |
| % Faculty Full-Time | 2.0% |
| Citations per Publication | 1.25% |
| Field-Weighted Citation Impact | 1.25% |
| Publications in Top 5% Journals | 1.0% |
| Publications in Top 25% Journals | 0.5% |

Each factor is standardized (population z-score) across all 434 schools,
multiplied by its weight, summed, rescaled so the lowest school is 0, and
scaled to a 0–100 "Overall Score." Ranks are assigned by that rounded score
(ties share a rank, USNWR-style). Schools missing SAT/ACT data have that
5% weight folded into the graduation-rate factor (16% -> 21%), matching
USNWR's test-optional handling.

## Key finding: what moves UVA's rank

Running the model on UVA (baseline: score 82.1, rank ~25 in this replica)
against every factor with an equal, apples-to-apples "+1 population std
dev" bump shows the highest-leverage levers, in order:

1. **Peer Assessment Score** (20% weight) — the single biggest lever by a
   wide margin. UVA already scores well here (top quartile), but because
   the weight is so large, even a modest +0.1–0.2 move in the average
   reputational survey score (out of 5) is worth 4–6 rank positions.
2. **Six-Year Graduation Rate** (16% weight) — UVA is already strong
   (~95%), but this factor's sheer weight means +1–2 points is still worth
   ~4 rank positions.
3. **Graduation Rate Performance** (10% weight) — UVA's actual six-year
   grad rate is currently almost exactly at its predicted rate (given
   selectivity/Pell mix), i.e. z ≈ 0. This is the most "underexploited"
   heavily-weighted factor: any actual-rate improvement here is nearly pure
   upside because there's no ceiling effect yet.
4. **Pell Grad Rate Performance** (5.5% weight) — UVA's weakest factor in
   z-score terms (below the population average): the gap between Pell and
   non-Pell six-year graduation rates is wider at UVA than at most peers.
   Closing this gap is both a rank lever and an equity outcome.
5. **Financial Resources per Student** and **Faculty Compensation** — real
   dollars move these directly and predictably, but the raw dollar amounts
   needed (thousands to tens of thousands per student/faculty member) are
   the least within a university's short-term control of the levers above.

Realistic, small combined moves (peer +0.1, grad rate +1pt, student/faculty
ratio -1) are enough to move UVA from rank ~25 to ~19 in this replica —
see `scenarios.py` output for the full comparison table.

