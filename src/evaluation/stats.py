"""
stats.py — Statistical significance testing

Uses Wilcoxon signed-rank test (non-parametric, appropriate for small n).
All pairwise comparisons against a reference method.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from typing import Dict
from src.evaluation.evaluator import AgentResult


def pairwise_significance(
    results: Dict[str, AgentResult],
    reference: str = 'Traditional Opt.',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each non-reference agent, run a Wilcoxon signed-rank test comparing
    per-episode profits against the reference agent.

    Returns a DataFrame with p-values and significance flags.
    """
    ref = results.get(reference)
    if ref is None:
        raise KeyError(f"Reference agent '{reference}' not in results.")

    ref_profits = np.array([e.total_profit for e in ref.episodes])
    rows = []

    for name, agent in results.items():
        if name == reference:
            continue

        agent_profits = np.array([e.total_profit for e in agent.episodes])
        n = min(len(ref_profits), len(agent_profits))

        # Wilcoxon requires paired samples of equal length
        try:
            stat, p = wilcoxon(agent_profits[:n], ref_profits[:n])
        except ValueError:
            # All differences are zero (identical sequences)
            stat, p = 0.0, 1.0

        diff_mean = float(np.mean(agent_profits[:n] - ref_profits[:n]))
        rows.append({
            'Method':        name,
            'vs Reference':  reference,
            'Mean Δ Profit': round(diff_mean, 1),
            'Wilcoxon stat': round(stat, 3),
            'p-value':       round(p, 4),
            'Significant':   p < alpha,
            'Direction':     'better' if diff_mean > 0 else 'worse',
        })

    return pd.DataFrame(rows)


def effect_size_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two paired arrays."""
    diff = a - b
    return float(np.mean(diff) / (np.std(diff) + 1e-9))


def robustness_table(
    results_by_elasticity: Dict[float, Dict[str, AgentResult]],
    reference: str = 'Traditional Opt.',
) -> pd.DataFrame:
    """
    Build a robustness table: rows = elasticity values, cols = agents.
    Cell = mean profit as % of the same agent's performance at training elasticity.
    """
    train_eps = -0.8  # assumed training elasticity

    base = {name: results_by_elasticity[train_eps][name].mean_profit
            for name in results_by_elasticity[train_eps]}

    rows = []
    for eps, res in sorted(results_by_elasticity.items()):
        row = {'Elasticity': eps}
        for name, agent_res in res.items():
            pct = (agent_res.mean_profit / base[name]) * 100 if base[name] != 0 else np.nan
            row[name] = round(pct, 1)
        rows.append(row)

    return pd.DataFrame(rows)
