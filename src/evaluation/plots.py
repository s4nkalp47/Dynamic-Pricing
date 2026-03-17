"""
plots.py — All paper figures in one place

Figures produced:
  fig1_performance_comparison   — revenue / profit / occupancy bars (like paper Fig 1)
  fig2_demand_stratified        — profit by demand level + price strategy
  fig3_learning_curves          — training profit over timesteps per variant
  fig4_robustness_heatmap       — % performance retained across elasticity values
  fig5_ablation                 — reward variant ablation bar chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional
from src.evaluation.evaluator import AgentResult, DEMAND_BUCKETS


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    'DQN (profit only)':         '#2ecc71',
    'DQN (occ cap)':             '#27ae60',
    'Double DQN':                '#3498db',
    'Dueling DQN':               '#2980b9',
    'Traditional Opt.':          '#e67e22',
    'Fixed Match':               '#e74c3c',
    'Fixed Undercut':            '#c0392b',
}

def _colour(name: str) -> str:
    for k, v in PALETTE.items():
        if k in name:
            return v
    return '#95a5a6'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


# ---------------------------------------------------------------------------
def fig1_performance_comparison(
    results: Dict[str, AgentResult],
    save_path: str = 'results/fig1_performance.png',
):
    """Three-panel bar chart: revenue, profit (with CI), occupancy."""
    names     = list(results.keys())
    revenues  = [r.mean_revenue   for r in results.values()]
    profits   = [r.mean_profit    for r in results.values()]
    profit_ci = [r.ci95           for r in results.values()]
    occs      = [r.mean_occupancy * 100 for r in results.values()]
    colours   = [_colour(n) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Performance Comparison Across All Methods', fontsize=15, fontweight='bold')

    def _bar(ax, values, ylabel, title, errors=None):
        x = np.arange(len(names))
        if errors:
            yerr = np.array([[p - lo, hi - p] for p, (lo, hi) in zip(values, errors)]).T
            ax.bar(x, values, color=colours, alpha=0.8, edgecolor='black',
                   linewidth=1, yerr=yerr, capsize=4, error_kw={'linewidth': 1.5})
        else:
            ax.bar(x, values, color=colours, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        for i, v in enumerate(values):
            ax.text(i, v * 1.01, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

    _bar(axes[0], revenues,  'Total Revenue (INR)',    'Revenue',  None)
    _bar(axes[1], profits,   'Total Profit (INR)',     'Profit\n(± 95% CI)', profit_ci)
    _bar(axes[2], occs,      'Average Occupancy (%)',  'Occupancy', None)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
def fig2_demand_stratified(
    results: Dict[str, AgentResult],
    highlight: List[str] = None,
    save_path: str = 'results/fig2_demand_stratified.png',
):
    """Profit by demand bucket + price ratio by demand bucket."""
    if highlight is None:
        highlight = ['DQN (profit only)', 'Traditional Opt.']

    buckets = list(DEMAND_BUCKETS.keys())
    x = np.arange(len(buckets))
    width = 0.8 / max(len(highlight), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Demand-Stratified Analysis', fontsize=15, fontweight='bold')

    # Profit by demand level
    ax = axes[0]
    for i, name in enumerate(highlight):
        if name not in results:
            continue
        r = results[name]
        # average per-bucket profit across all episodes
        bucket_profits = []
        for bucket in buckets:
            vals = [ep.profits_by_demand.get(bucket, 0.0) for ep in r.episodes]
            bucket_profits.append(np.mean(vals))
        ax.bar(x + i * width, bucket_profits, width=width * 0.9,
               label=name, color=_colour(name), alpha=0.8, edgecolor='black')
        for j, v in enumerate(bucket_profits):
            ax.text(x[j] + i * width, v + 20, f'{v:.0f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(highlight) - 1) / 2)
    ax.set_xticklabels([b.replace('_', '\n') for b in buckets])
    ax.set_ylabel('Avg Profit per Step (INR)', fontweight='bold')
    ax.set_title('(a) Profit by Demand Level', fontweight='bold')
    ax.legend(fontsize=9)

    # Price ratio by demand level — needs per-step data; approximate from
    # stored episode stats: here we compute episode-mean price ratio
    # We can only produce this if agents store per-step data;
    # as a fallback show overall mean price ratio as horizontal lines
    ax = axes[1]
    for name in highlight:
        if name not in results:
            continue
        r = results[name]
        ratio = r.mean_price_ratio
        ax.axhline(ratio, color=_colour(name), linewidth=2, linestyle='--',
                   label=f'{name}: {ratio:.3f}×')

    ax.set_xlabel('Demand Level')
    ax.set_ylabel('Average Price Ratio (× competitor)', fontweight='bold')
    ax.set_title('(b) Overall Price Ratio', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.8, 1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
def fig3_learning_curves(
    curves: Dict[str, np.ndarray],
    window: int = 5000,
    save_path: str = 'results/fig3_learning_curves.png',
):
    """
    curves: dict of {agent_name: array of per-step profits from training}
    Produces smoothed learning curves.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Training Learning Curves (Smoothed)', fontweight='bold')
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Smoothed Profit per Step (INR)', fontweight='bold')

    for name, data in curves.items():
        smoothed = pd.Series(data).rolling(window, min_periods=1).mean().values
        steps    = np.arange(len(smoothed))
        ax.plot(steps, smoothed, label=name, color=_colour(name), linewidth=2)

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
def fig4_robustness_heatmap(
    robustness_df: pd.DataFrame,
    save_path: str = 'results/fig4_robustness.png',
):
    """
    robustness_df: rows = elasticity, cols = agent names, values = % retained
    """
    agents = [c for c in robustness_df.columns if c != 'Elasticity']
    eps    = robustness_df['Elasticity'].values
    matrix = robustness_df[agents].values.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 1.5), 4))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=60, vmax=105, aspect='auto')

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(eps)))
    ax.set_yticklabels([f'ε={e}' for e in eps], fontsize=10)
    ax.set_title('Robustness to Demand Misspecification\n(% of training-elasticity performance retained)',
                 fontweight='bold')

    for i in range(len(eps)):
        for j in range(len(agents)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    color='black', fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax, label='% performance retained')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
def fig5_ablation(
    ablation_results: Dict[str, AgentResult],
    reference_profit: float,
    save_path: str = 'results/fig5_ablation.png',
):
    """Bar chart comparing reward variant ablation."""
    names   = list(ablation_results.keys())
    profits = [r.mean_profit for r in ablation_results.values()]
    cis     = [r.ci95        for r in ablation_results.values()]
    colours = ['#3498db', '#27ae60', '#e67e22', '#9b59b6'][:len(names)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    yerr = np.array([[p - lo, hi - p] for p, (lo, hi) in zip(profits, cis)]).T

    ax.bar(x, profits, color=colours, alpha=0.8, edgecolor='black',
           linewidth=1.2, yerr=yerr, capsize=5)
    ax.axhline(reference_profit, color='black', linestyle='--',
               linewidth=1.5, label=f'Traditional Opt. ({reference_profit:.0f})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Mean Profit (INR)', fontweight='bold')
    ax.set_title('Reward Shaping Ablation\n(all variants: same network, same data)',
                 fontweight='bold')
    ax.legend(fontsize=10)
    for i, v in enumerate(profits):
        ax.text(i, v + max(profits) * 0.01, f'{v:.0f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
