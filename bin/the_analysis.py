#!/usr/bin/env python3
"""
Why is UVA's THE World University Ranking Declining?

Analysis of Times Higher Education (THE) World University Rankings,
examining UVA's score and rank trends, benchmarking against Public AAU
peers, and looking at global (especially Chinese) competitive trends.

This script is a non-interactive (headless) conversion of THE_notebook.ipynb.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend -- no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
# This file lives at <project_root>/bin/the_analysis.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Prefer lowercase 'data' but fall back to 'Data' if needed
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "Data"

PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 250)


def savefig(fig: plt.Figure, name: str) -> None:
    """Save a figure to PLOTS_DIR and close it."""
    out_path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    logger.info("Loading data from %s", DATA_DIR / "THE_ALL.csv")
    df = pd.read_csv(DATA_DIR / "THE_ALL.csv")
    return df


# --------------------------------------------------------------------------
# 1. UVA Rank vs. Score Over Time
# --------------------------------------------------------------------------
def uva_rank_vs_score(df: pd.DataFrame) -> pd.DataFrame:
    uva = df[df["name"] == "University of Virginia"].sort_values("year")
    uva = uva[
        [
            "year",
            "rank",
            "scores_overall",
            "scores_teaching",
            "scores_research",
            "scores_citations",
            "scores_industry_income",
            "scores_international_outlook",
        ]
    ]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(uva["year"], uva["scores_overall"], color="tab:blue", marker="o", label="Overall Score")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Overall Score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(uva["year"], uva["rank"], color="tab:red", marker="s", label="Rank")
    ax2.set_ylabel("Rank (lower is better)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.invert_yaxis()

    ax1.set_title("University of Virginia: Overall Score vs. Rank, 2011-2026")
    savefig(fig, "01_uva_score_vs_rank.png")

    return uva


# --------------------------------------------------------------------------
# 2. Sub-score Trends for UVA
# --------------------------------------------------------------------------
def uva_subscore_trends(uva: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in [
        ("scores_teaching", "Teaching"),
        ("scores_research", "Research"),
        ("scores_citations", "Citations"),
        ("scores_industry_income", "Industry Income"),
        ("scores_international_outlook", "International Outlook"),
    ]:
        ax.plot(uva["year"], uva[col], marker="o", label=label)

    ax.set_xlabel("Year")
    ax.set_ylabel("Score")
    ax.set_title("UVA Sub-score Trends")
    ax.legend()
    savefig(fig, "02_uva_subscore_trends.png")


# --------------------------------------------------------------------------
# 3. Benchmarking Against Public AAU Peers (2020 vs 2026)
# --------------------------------------------------------------------------
def aau_benchmarking(df: pd.DataFrame) -> pd.DataFrame:
    recent = df[df["year"].isin([2020, 2026])]
    aau = recent[recent["public_aau_flag"] == 1]

    piv = aau.pivot_table(index="name", columns="year", values=["rank", "scores_overall"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.dropna(subset=["rank_2020", "rank_2026"])

    piv["rank_change"] = piv["rank_2020"] - piv["rank_2026"]  # positive = improved
    piv["score_change"] = piv["scores_overall_2026"] - piv["scores_overall_2020"]
    piv = piv.sort_values("rank_change")

    logger.info(
        "Mean score change across Public AAU peers: %s",
        round(piv["score_change"].mean(), 2),
    )
    logger.info(
        "Mean rank change across Public AAU peers: %s",
        round(piv["rank_change"].mean(), 2),
    )
    if "University of Virginia" in piv.index:
        logger.info("UVA score change: %s", round(piv.loc["University of Virginia", "score_change"], 2))
        logger.info("UVA rank change: %s", round(piv.loc["University of Virginia", "rank_change"], 2))

    # Scatter: score change vs rank change
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ["tab:red" if name == "University of Virginia" else "tab:blue" for name in piv.index]
    sizes = [80 if name == "University of Virginia" else 30 for name in piv.index]

    ax.scatter(piv["score_change"], piv["rank_change"], c=colors, s=sizes)

    for name, row in piv.iterrows():
        if name == "University of Virginia" or abs(row["rank_change"]) > 35 or abs(row["score_change"]) > 5:
            ax.annotate(
                name,
                (row["score_change"], row["rank_change"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Change in Overall Score, 2020 -> 2026")
    ax.set_ylabel("Change in Rank, 2020 -> 2026 (positive = improved)")
    ax.set_title("Public AAU Schools: Score Change vs. Rank Change (UVA in red)")
    savefig(fig, "03_aau_score_vs_rank_change_scatter.png")

    # Horizontal bar: rank change
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = range(len(piv))
    colors = ["tab:red" if name == "University of Virginia" else "tab:blue" for name in piv.index]

    ax.barh(y_pos, piv["rank_change"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Rank Change 2020->2026 (positive = improved rank)")
    ax.set_title("Public AAU Schools: Rank Change 2020-2026 (UVA in red)")
    savefig(fig, "04_aau_rank_change_barh.png")

    return piv


# --------------------------------------------------------------------------
# 4. The Rising Bar: How Many Schools Now Beat UVA's Score?
# --------------------------------------------------------------------------
def rising_bar(df: pd.DataFrame, uva: pd.DataFrame) -> float:
    uva_2026_score = uva.loc[uva["year"] == 2026, "scores_overall"].values[0]

    counts = {}
    for yr in [2020, 2026]:
        sub = df[df["year"] == yr]
        counts[yr] = {
            "total_schools": sub.shape[0],
            "schools_above_uva_score": (sub["scores_overall"] > uva_2026_score).sum(),
        }

    counts_df = pd.DataFrame(counts).T
    logger.info("Institution counts vs UVA 2026 score (%s):\n%s", uva_2026_score, counts_df)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["2020", "2026"], counts_df["schools_above_uva_score"], color=["tab:blue", "tab:red"])
    ax.set_ylabel("# of institutions worldwide")
    ax.set_title(f"Number of Institutions Scoring Above UVA's 2026 Score ({uva_2026_score})")
    for i, v in enumerate(counts_df["schools_above_uva_score"]):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    savefig(fig, "05_institutions_above_uva_score.png")

    return uva_2026_score


# --------------------------------------------------------------------------
# 5. Global Country Trends: Top-200 Representation
# --------------------------------------------------------------------------
def global_country_trends(df: pd.DataFrame) -> None:
    top200_2020 = df[(df["year"] == 2020) & (df["rank"] <= 200)]["location"].value_counts().head(10)
    top200_2026 = df[(df["year"] == 2026) & (df["rank"] <= 200)]["location"].value_counts().head(10)

    country_compare = pd.DataFrame({"2020": top200_2020, "2026": top200_2026}).fillna(0)
    country_compare = country_compare.sort_values("2020", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    country_compare.plot(kind="bar", ax=ax)
    ax.set_ylabel("# of Universities in Global Top 200")
    ax.set_title("Top-200 Representation by Country: 2020 vs. 2026")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    savefig(fig, "06_top200_representation_by_country.png")


# --------------------------------------------------------------------------
# 6. China's Rapid Score Gains (2020 -> 2026)
# --------------------------------------------------------------------------
def china_score_gains(df: pd.DataFrame, uva_2026_score: float) -> None:
    china = df[(df["location"] == "China") & (df["year"].isin([2020, 2026]))]
    cpiv = china.pivot_table(index="name", columns="year", values=["rank", "scores_overall"], aggfunc="first")
    cpiv.columns = [f"{a}_{b}" for a, b in cpiv.columns]
    cpiv = cpiv.dropna(subset=["rank_2020", "rank_2026"])
    cpiv["score_change"] = cpiv["scores_overall_2026"] - cpiv["scores_overall_2020"]
    cpiv = cpiv.sort_values("score_change", ascending=False)

    top_china = cpiv.head(7)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(top_china))
    width = 0.35

    ax.bar([i - width / 2 for i in x], top_china["scores_overall_2020"], width, label="2020 Score")
    ax.bar([i + width / 2 for i in x], top_china["scores_overall_2026"], width, label="2026 Score")

    ax.axhline(uva_2026_score, color="tab:red", linestyle="--", label=f"UVA 2026 Score ({uva_2026_score})")

    ax.set_xticks(list(x))
    ax.set_xticklabels(top_china.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Overall Score")
    ax.set_title("Leading Chinese Universities: Score Gains 2020-2026 vs. UVA")
    ax.legend()
    savefig(fig, "07_china_score_gains.png")


# --------------------------------------------------------------------------
# 7. Hong Kong Universities: Score and Rank Trends
# --------------------------------------------------------------------------
def hong_kong_trends(df: pd.DataFrame, uva_2026_score: float) -> None:
    hk_schools = [
        "University of Hong Kong",
        "The Hong Kong University of Science and Technology",
        "City University of Hong Kong",
        "The Chinese University of Hong Kong",
        "The Hong Kong Polytechnic University",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in hk_schools:
        sub = df[df["name"] == name].sort_values("year")
        ax.plot(sub["year"], sub["scores_overall"], marker="o", label=name)

    ax.axhline(uva_2026_score, color="gray", linestyle="--", label=f"UVA 2026 ({uva_2026_score})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Overall Score")
    ax.set_title("Hong Kong Universities: Overall Score, 2011-2026")
    ax.legend(fontsize=8)
    savefig(fig, "08_hong_kong_score_trends.png")

    hk = df[(df["location"] == "Hong Kong") & (df["year"].isin([2020, 2026]))]
    hkpiv = hk.pivot_table(index="name", columns="year", values=["rank", "scores_overall"], aggfunc="first")
    hkpiv.columns = [f"{a}_{b}" for a, b in hkpiv.columns]
    hkpiv = hkpiv.dropna(subset=["rank_2020", "rank_2026"])
    hkpiv["rank_change"] = hkpiv["rank_2020"] - hkpiv["rank_2026"]
    hkpiv["score_change"] = hkpiv["scores_overall_2026"] - hkpiv["scores_overall_2020"]
    hkpiv = hkpiv.sort_values("score_change", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(hkpiv))
    width = 0.35

    ax.bar([i - width / 2 for i in x], hkpiv["scores_overall_2020"], width, label="2020 Score")
    ax.bar([i + width / 2 for i in x], hkpiv["scores_overall_2026"], width, label="2026 Score")
    ax.axhline(uva_2026_score, color="tab:red", linestyle="--", label=f"UVA 2026 Score ({uva_2026_score})")

    ax.set_xticks(list(x))
    ax.set_xticklabels(hkpiv.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Overall Score")
    ax.set_title("Hong Kong Universities: Score 2020 vs 2026 vs. UVA")
    ax.legend()
    savefig(fig, "09_hong_kong_score_2020_vs_2026.png")

    for yr in [2020, 2026]:
        n = ((df["year"] == yr) & (df["rank"] <= 200) & (df["location"] == "Hong Kong")).sum()
        logger.info("Hong Kong universities in global Top 200 (%s): %s", yr, n)


# --------------------------------------------------------------------------
# 8. UVA Sub-score Distribution vs. Public AAU Peers (2026)
# --------------------------------------------------------------------------
def uva_subscore_distribution(df: pd.DataFrame) -> None:
    aau_2026 = df[(df["public_aau_flag"] == 1) & (df["year"] == 2026)].copy()
    uva_2026 = aau_2026[aau_2026["name"] == "University of Virginia"].iloc[0]

    pillars = [
        ("scores_overall", "Overall"),
        ("scores_teaching", "Teaching"),
        ("scores_research", "Research"),
        ("scores_citations", "Citations"),
        ("scores_industry_income", "Industry Income"),
        ("scores_international_outlook", "International Outlook"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    rows = []
    for ax, (col, label) in zip(axes, pillars):
        vals = aau_2026[col].dropna().values
        uva_val = uva_2026[col]

        bp = ax.boxplot(
            vals,
            vert=True,
            patch_artist=True,
            widths=0.4,
            boxprops=dict(facecolor="steelblue", alpha=0.4),
            medianprops=dict(color="steelblue", linewidth=2),
            whiskerprops=dict(color="steelblue"),
            capprops=dict(color="steelblue"),
            flierprops=dict(marker=""),
        )

        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(1 + jitter, vals, color="steelblue", alpha=0.6, zorder=3, s=30)

        ax.scatter(1, uva_val, color="tab:red", zorder=5, s=100, label=f"UVA ({uva_val})")
        ax.axhline(uva_val, color="tab:red", linestyle="--", alpha=0.5)

        pct = (vals < uva_val).sum() / len(vals) * 100
        rank_n = int((vals >= uva_val).sum())

        ax.set_title(f"{label}\nUVA: {uva_val} | Rank {rank_n}/{len(vals)} peers | {pct:.0f}th pctile", fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc="lower right")

        rows.append(
            {
                "Pillar": label,
                "UVA Score": uva_val,
                "Peer Median": round(float(np.median(vals)), 1),
                "Peer Min": round(float(vals.min()), 1),
                "Peer Max": round(float(vals.max()), 1),
                "UVA Rank (of 27)": int((vals >= uva_val).sum()),
                "UVA Percentile": f"{(vals < uva_val).sum() / len(vals) * 100:.0f}th",
            }
        )

    fig.suptitle(
        "UVA Sub-score Distribution vs. Public AAU Peers (2026)\n"
        "(red dot = UVA, blue = peer distribution)",
        fontsize=12,
    )
    savefig(fig, "10_uva_subscore_distribution_vs_aau.png")

    summary_df = pd.DataFrame(rows).set_index("Pillar")
    logger.info("Summary table: UVA score, peer median, peer rank, percentile:\n%s", summary_df)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    df = load_data()

    uva = uva_rank_vs_score(df)
    uva_subscore_trends(uva)
    aau_benchmarking(df)
    uva_2026_score = rising_bar(df, uva)
    global_country_trends(df)
    china_score_gains(df, uva_2026_score)
    hong_kong_trends(df, uva_2026_score)
    uva_subscore_distribution(df)

    logger.info("All plots written to: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
