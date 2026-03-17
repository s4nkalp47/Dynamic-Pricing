"""
baselines.py — Deterministic pricing baselines

TraditionalOptimizer  : L-BFGS-B on the known demand function (oracle)
FixedMatch            : always matches competitor price
FixedUndercut         : always undercuts competitor by a fixed fraction
"""

import numpy as np
from scipy.optimize import minimize_scalar


class TraditionalOptimizer:
    """
    Analytic profit maximiser with *perfect* knowledge of the demand function.

    This is the oracle baseline — it represents the ceiling for any method
    that assumes the demand model is correctly specified.  It deliberately
    does NOT add any RL-style exploration or uncertainty handling so that
    the comparison is fair.
    """

    def __init__(self, env):
        self.base_adr       = env.base_adr
        self.base_demand    = env.base_demand
        self.elasticity     = env.elasticity
        self.cost_ratio     = env.cost_ratio
        self.hotel_capacity = env.hotel_capacity

    def _profit(self, price, comp_price, demand_idx, holiday):
        price_ratio  = price / comp_price
        elas_effect  = 1.0 + self.elasticity * (price_ratio - 1.0)
        market_mult  = 0.6 + 0.8 * demand_idx
        holiday_mult = 1.15 if holiday else 1.0

        demand = np.clip(
            self.base_demand * elas_effect * market_mult * holiday_mult,
            0, self.hotel_capacity
        )
        cost   = self.base_adr * self.cost_ratio
        return (price - cost) * demand

    def get_price(self, comp_price: float, demand_idx: float, holiday: float) -> float:
        """Return profit-maximising price via bounded scalar optimisation."""
        result = minimize_scalar(
            lambda p: -self._profit(p, comp_price, demand_idx, holiday),
            bounds=(comp_price * 0.5, comp_price * 2.0),
            method='bounded',
        )
        return float(result.x)


class FixedMatch:
    """Always prices at exactly the competitor price."""

    def get_price(self, comp_price: float, **_) -> float:
        return float(comp_price)


class FixedUndercut:
    """Always prices at (1 - fraction) × competitor price."""

    def __init__(self, fraction: float = 0.05):
        self.fraction = fraction

    def get_price(self, comp_price: float, **_) -> float:
        return float(comp_price * (1.0 - self.fraction))
