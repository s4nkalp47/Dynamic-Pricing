"""
evaluator.py — Shared evaluation engine

All agents (DQN variants + baselines) run on *identical* decision sequences
so results are directly comparable.  Per-episode stats are stored so we can
compute confidence intervals and run significance tests downstream.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    total_profit:   float
    total_revenue:  float
    avg_occupancy:  float
    avg_price_ratio: float
    profits_by_demand: Dict[str, float]  # keyed by demand bucket label


@dataclass
class AgentResult:
    name:     str
    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def mean_profit(self)   -> float: return float(np.mean([e.total_profit   for e in self.episodes]))
    @property
    def std_profit(self)    -> float: return float(np.std( [e.total_profit   for e in self.episodes]))
    @property
    def mean_revenue(self)  -> float: return float(np.mean([e.total_revenue  for e in self.episodes]))
    @property
    def mean_occupancy(self)-> float: return float(np.mean([e.avg_occupancy  for e in self.episodes]))
    @property
    def mean_price_ratio(self)->float:return float(np.mean([e.avg_price_ratio for e in self.episodes]))
    @property
    def ci95(self) -> tuple:
        """95% confidence interval on mean profit (normal approx)."""
        n   = len(self.episodes)
        sem = self.std_profit / np.sqrt(n)
        return (self.mean_profit - 1.96 * sem,
                self.mean_profit + 1.96 * sem)


# ---------------------------------------------------------------------------
DEMAND_BUCKETS = {
    'very_low':  (0.0,  0.2),
    'low':       (0.2,  0.4),
    'medium':    (0.4,  0.6),
    'high':      (0.6,  0.8),
    'very_high': (0.8,  1.01),
}


def _demand_bucket(demand_idx: float) -> str:
    for label, (lo, hi) in DEMAND_BUCKETS.items():
        if lo <= demand_idx < hi:
            return label
    return 'very_high'


# ---------------------------------------------------------------------------
def run_episode(env, agent, deterministic: bool = True) -> EpisodeResult:
    """
    Run one episode and return per-episode metrics.

    agent must expose either:
      .predict(obs, deterministic=True)  →  (action, _state)   [SB3 RL model]
    or:
      .get_price(comp_price, demand_idx, holiday)               [baseline]
    """
    obs, _ = env.reset()
    total_profit  = 0.0
    total_revenue = 0.0
    occupancies   = []
    price_ratios  = []
    demand_profits: Dict[str, List[float]] = {k: [] for k in DEMAND_BUCKETS}

    is_rl = hasattr(agent, 'predict')

    while True:
        row        = env.df.iloc[env.current_step]
        comp_price = float(row['competitor_price'])
        demand_idx = float(row['demand_index'])
        holiday    = float(row['holiday_flag'])

        if is_rl:
            action, _ = agent.predict(obs, deterministic=deterministic)
            action    = int(np.squeeze(action))
        else:
            opt_price = agent.get_price(comp_price=comp_price,
                                        demand_idx=demand_idx,
                                        holiday=holiday)
            # Map continuous price → nearest discrete action (fair comparison)
            ratio  = opt_price / comp_price
            action = int(np.argmin(np.abs(env.price_multipliers - ratio)))

        obs, _, terminated, truncated, info = env.step(action)

        total_profit  += info['profit']
        total_revenue += info['price'] * info['demand']
        occupancies.append(info['occupancy'])
        price_ratios.append(info['price_ratio'])

        bucket = _demand_bucket(demand_idx)
        demand_profits[bucket].append(info['profit'])

        if terminated or truncated:
            break

    return EpisodeResult(
        total_profit    = total_profit,
        total_revenue   = total_revenue,
        avg_occupancy   = float(np.mean(occupancies)),
        avg_price_ratio = float(np.mean(price_ratios)),
        profits_by_demand = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in demand_profits.items()
        },
    )


def evaluate_agent(env, agent, name: str, n_episodes: int = 20) -> AgentResult:
    """Evaluate a single agent over n_episodes and return AgentResult."""
    result = AgentResult(name=name)
    for ep in range(n_episodes):
        ep_result = run_episode(env, agent)
        result.episodes.append(ep_result)
        if (ep + 1) % 5 == 0:
            print(f"  [{name}] episode {ep+1}/{n_episodes} | "
                  f"profit={ep_result.total_profit:.0f} INR")
    return result


def evaluate_all(env, agents: Dict[str, Any], n_episodes: int = 20) -> Dict[str, AgentResult]:
    """
    Evaluate multiple agents.

    agents: dict of {display_name: agent_object}
    """
    results = {}
    for name, agent in agents.items():
        print(f"\n→ Evaluating: {name}")
        results[name] = evaluate_agent(env, agent, name, n_episodes)
    return results


# ---------------------------------------------------------------------------
def results_to_dataframe(results: Dict[str, AgentResult],
                         reference: str = 'Traditional Opt.') -> pd.DataFrame:
    """
    Convert AgentResult dict to a summary DataFrame with CI and vs-reference %.
    """
    ref_profit = results[reference].mean_profit if reference in results else None

    rows = []
    for name, r in results.items():
        lo, hi = r.ci95
        vs_ref = ((r.mean_profit - ref_profit) / abs(ref_profit) * 100
                  if ref_profit else float('nan'))
        rows.append({
            'Method':          name,
            'Mean Profit':     round(r.mean_profit, 1),
            'Std Profit':      round(r.std_profit, 1),
            'CI95 Low':        round(lo, 1),
            'CI95 High':       round(hi, 1),
            'Mean Revenue':    round(r.mean_revenue, 1),
            'Mean Occupancy':  round(r.mean_occupancy * 100, 2),
            'Mean Price Ratio':round(r.mean_price_ratio, 4),
            'vs Reference (%)':round(vs_ref, 2),
        })
    return pd.DataFrame(rows)
