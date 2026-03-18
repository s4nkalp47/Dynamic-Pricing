"""
hotel_env.py — Clean Hotel Pricing Environment

Performance fix: all dataset columns preloaded as numpy arrays.
Eliminates pandas .iloc row lookup overhead which was bottlenecking
CPU→GPU throughput (GPU was sitting at 25-48% utilization).

Other features:
  - normalise_reward=True: divides reward by running std (Welford algorithm)
  - state_variant='8d': adds lead_time, days_until_event, customer_rating
  - reward_variant='profit_with_occ_cap': occupancy penalty (ablation only)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional

REWARD_VARIANTS = {'profit_only', 'profit_with_occ_cap'}
STATE_VARIANTS  = {'5d', '8d'}


class HotelPricingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str,
        reward_variant: str    = 'profit_only',
        state_variant: str     = '5d',
        normalise_reward: bool = False,
        occ_cap: float         = 0.85,
        occ_penalty_weight: float = 0.1,
        episode_length: int    = 1000,
        seed: Optional[int]    = None,
    ):
        super().__init__()

        if reward_variant not in REWARD_VARIANTS:
            raise ValueError(f"reward_variant must be one of {REWARD_VARIANTS}")
        if state_variant not in STATE_VARIANTS:
            raise ValueError(f"state_variant must be one of {STATE_VARIANTS}")

        self.df              = pd.read_csv(data_path).dropna().reset_index(drop=True)
        self.state_variant   = state_variant
        self.reward_variant  = reward_variant
        self.normalise_reward = normalise_reward
        self._validate_data()

        self.episode_length = min(episode_length, len(self.df) - 1)
        self.occ_cap        = occ_cap
        n_obs               = 5 if state_variant == '5d' else 8

        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(n_obs,), dtype=np.float32)
        self.n_actions         = 25
        self.price_multipliers = np.linspace(0.80, 1.40, self.n_actions)
        self.action_space      = spaces.Discrete(self.n_actions)

        # Market parameters
        self.base_adr        = float(self.df['adr'].mean())
        self.base_demand     = float(self.df['occupancy'].mean() * 100)
        self.elasticity      = -0.8
        self.cost_ratio      = 0.60
        self.hotel_capacity  = 100

        typical_profit         = (self.base_adr * (1 - self.cost_ratio)) * self.base_demand
        self.occ_penalty_scale = occ_penalty_weight * typical_profit

        # ---------------------------------------------------------------
        # PERFORMANCE FIX: preload all columns as numpy arrays
        # Eliminates pandas .iloc overhead on every env step.
        # This is the main CPU bottleneck at ~25-48% GPU utilization.
        # ---------------------------------------------------------------
        self._arr_competitor_price = self.df['competitor_price'].to_numpy(dtype=np.float32)
        self._arr_demand_index     = self.df['demand_index'].to_numpy(dtype=np.float32)
        self._arr_holiday_flag     = self.df['holiday_flag'].to_numpy(dtype=np.float32)
        self._arr_day_of_week      = self.df['day_of_week'].to_numpy(dtype=np.float32)
        self._arr_occupancy        = self.df['occupancy'].to_numpy(dtype=np.float32)

        if state_variant == '8d':
            self._arr_lead_time        = self.df['lead_time'].to_numpy(dtype=np.float32)
            self._arr_days_until_event = self.df['days_until_event'].to_numpy(dtype=np.float32)
            self._arr_customer_rating  = self.df['customer_rating'].to_numpy(dtype=np.float32)

        # Reward normalisation state (Welford online algorithm)
        self._reward_count = 0
        self._reward_mean  = 0.0
        self._reward_M2    = 0.0

        self.current_step: int = 0
        self._rng = np.random.default_rng(seed)

        print(f"✓ HotelPricingEnv | state={state_variant} ({n_obs}D) | "
              f"reward={reward_variant} | normalise={normalise_reward} | "
              f"rows={len(self.df):,}")

    # ---------------------------------------------------------------- validate
    def _validate_data(self):
        required = ['adr', 'occupancy', 'competitor_price',
                    'demand_index', 'holiday_flag', 'day_of_week']
        if self.state_variant == '8d':
            required += ['lead_time', 'days_until_event', 'customer_rating']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Data missing columns: {missing}")
        if self.df['competitor_price'].max() < 100:
            raise ValueError("competitor_price looks normalised — pass denormalised CSV")

    # --------------------------------------------------------- reward stats
    def _update_reward_stats(self, r: float):
        self._reward_count += 1
        delta              = r - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_M2   += delta * (r - self._reward_mean)

    def _reward_std(self) -> float:
        if self._reward_count < 2:
            return 1.0
        return float(np.sqrt(self._reward_M2 / (self._reward_count - 1)) + 1e-8)

    # ------------------------------------------------------------------ obs
    def _get_obs(self) -> np.ndarray:
        i         = self.current_step
        comp_norm = np.clip(self._arr_competitor_price[i] / self.base_adr, 0.0, 1.0)

        obs = [
            np.clip(self._arr_demand_index[i], 0.0, 1.0),
            np.clip(self._arr_occupancy[i],    0.0, 1.0),
            self._arr_holiday_flag[i],
            np.clip(self._arr_day_of_week[i],  0.0, 1.0),
            comp_norm,
        ]

        if self.state_variant == '8d':
            obs += [
                np.clip(self._arr_lead_time[i],        0.0, 1.0),
                np.clip(self._arr_days_until_event[i], 0.0, 1.0),
                np.clip(self._arr_customer_rating[i],  0.0, 1.0),
            ]

        return np.array(obs, dtype=np.float32)

    # --------------------------------------------------------------- demand
    def compute_demand(self, price, comp_price, demand_idx, holiday) -> float:
        """Elasticity demand model. Public so baselines can call it directly."""
        price_ratio  = price / comp_price
        elas_effect  = 1.0 + self.elasticity * (price_ratio - 1.0)
        market_mult  = 0.6 + 0.8 * demand_idx
        holiday_mult = 1.15 if holiday else 1.0
        demand       = self.base_demand * elas_effect * market_mult * holiday_mult
        return float(np.clip(demand, 0.0, self.hotel_capacity))

    # ----------------------------------------------------------------- step
    def step(self, action: int):
        i          = self.current_step
        comp_price = float(self._arr_competitor_price[i])
        demand_idx = float(self._arr_demand_index[i])
        holiday    = float(self._arr_holiday_flag[i])

        price     = comp_price * self.price_multipliers[action]
        demand    = self.compute_demand(price, comp_price, demand_idx, holiday)
        cost      = self.base_adr * self.cost_ratio
        profit    = (price - cost) * demand
        occupancy = demand / self.hotel_capacity

        reward = profit
        if self.reward_variant == 'profit_with_occ_cap' and occupancy > self.occ_cap:
            excess  = occupancy - self.occ_cap
            reward -= self.occ_penalty_scale * (excess / (1.0 - self.occ_cap))

        if self.normalise_reward:
            self._update_reward_stats(reward)
            reward = reward / self._reward_std()

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        obs        = (self._get_obs() if not terminated
                      else np.zeros(self.observation_space.shape, dtype=np.float32))

        info = {
            'price':        price,
            'price_ratio':  price / comp_price,
            'comp_price':   comp_price,
            'demand':       demand,
            'profit':       profit,   # always raw INR
            'occupancy':    occupancy,
            'demand_index': demand_idx,
            'holiday':      holiday,
            'state_variant': self.state_variant,
        }
        return obs, reward, terminated, False, info

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start         = max(1, len(self.df) - self.episode_length)
        self.current_step = int(self._rng.integers(0, max_start))
        return self._get_obs(), {}

    # --------------------------------------------------------------- render
    def render(self):
        i = self.current_step
        print(f"Step {i:5d} | demand={self._arr_demand_index[i]:.2f} | "
              f"comp={self._arr_competitor_price[i]:.0f} INR | "
              f"holiday={int(self._arr_holiday_flag[i])}")