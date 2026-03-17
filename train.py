"""
train.py — Train DQN variants + PPO + SAC for hotel dynamic pricing

Algorithms:
  DQN (vanilla)  — standard DQN, discrete actions, lr=5e-4, 300k steps
  Double DQN     — lower lr=3e-4, 500k steps (stability fix)
  Dueling DQN    — wider net [512,512,256], lr=3e-4, 500k steps
  PPO            — on-policy, most stable, 500k steps
  SAC            — off-policy continuous, best for fine-grained pricing, 500k steps
                   (uses continuous action space wrapper)

Why PPO/SAC:
  DQN's off-policy + high reward variance = instability in this env.
  PPO is on-policy so gradients are always fresh — no stale replay issues.
  SAC has entropy regularisation which prevents overconfident pricing.

Usage:
    # Full pipeline (all 5 algorithms)
    python train.py --data data/enhanced_data_fixed.csv

    # Skip already-trained models
    python train.py --data data/enhanced_data_fixed.csv --skip vanilla double

    # DQN variants only (no PPO/SAC)
    python train.py --data data/enhanced_data_fixed.csv --dqn_only

    # PPO and SAC only
    python train.py --data data/enhanced_data_fixed.csv --ppo_sac_only
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.hotel_env import HotelPricingEnv


# ---------------------------------------------------------------------------
# Profit logger callback (works for all SB3 algorithms)
# ---------------------------------------------------------------------------
class ProfitLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_profits: list = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        if infos and 'profit' in infos[0]:
            self.step_profits.append(infos[0]['profit'])
        return True


# ---------------------------------------------------------------------------
# Continuous action wrapper for SAC
# SAC requires a continuous action space. We wrap the discrete env so that
# a continuous action in [-1, 1] is mapped to the nearest discrete price level.
# ---------------------------------------------------------------------------
class ContinuousPricingWrapper(gym.Wrapper):
    """
    Wraps HotelPricingEnv to expose a continuous Box action space for SAC.
    Action: scalar in [-1, 1] → mapped to price multiplier in [0.80, 1.40].
    """
    def __init__(self, env: HotelPricingEnv):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0],  dtype=np.float32),
        )
        self._multipliers = env.price_multipliers  # [0.80 .. 1.40]

    def step(self, action):
        # Map [-1,1] → [0.80, 1.40] → nearest discrete bin
        continuous = float(np.clip(action[0], -1.0, 1.0))
        multiplier = 0.80 + (continuous + 1.0) / 2.0 * (1.40 - 0.80)
        discrete   = int(np.argmin(np.abs(self._multipliers - multiplier)))
        return self.env.step(discrete)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def _make_dqn(env, lr: float, net_arch: list, log_dir: str) -> DQN:
    return DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=500,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        policy_kwargs={'net_arch': net_arch, 'activation_fn': torch.nn.ReLU},
        verbose=0,
        tensorboard_log=log_dir,
        device='auto',
    )


def _make_ppo(env, log_dir: str) -> PPO:
    return PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,        # steps per rollout per env
        batch_size=256,
        n_epochs=10,         # gradient passes per rollout
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # small entropy bonus → encourages exploration
        policy_kwargs={
            'net_arch': [256, 256, 128],
            'activation_fn': torch.nn.ReLU,
        },
        verbose=0,
        tensorboard_log=log_dir,
        device='auto',
    )


def _make_sac(env, log_dir: str) -> SAC:
    return SAC(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        ent_coef='auto',     # automatic entropy tuning (key SAC feature)
        policy_kwargs={
            'net_arch': [256, 256, 128],
            'activation_fn': torch.nn.ReLU,
        },
        verbose=0,
        tensorboard_log=log_dir,
        device='auto',
    )


# ---------------------------------------------------------------------------
# Generic training function
# ---------------------------------------------------------------------------
def train_one(
    data_path: str,
    algo: str,          # 'dqn_vanilla' | 'dqn_double' | 'dqn_dueling' | 'ppo' | 'sac'
    run_name: str,
    steps: int,
    seed: int = 42,
) -> tuple:

    log_dir   = f'logs/{run_name}'
    model_dir = f'models/{run_name}'
    os.makedirs(model_dir, exist_ok=True)

    # --- env factory -------------------------------------------------------
    def make_base_env():
        return HotelPricingEnv(
            data_path,
            reward_variant='profit_only',
            normalise_reward=True,
            seed=seed,
        )

    def make_env_dqn():
        return Monitor(make_base_env())

    def make_env_sac():
        return Monitor(ContinuousPricingWrapper(make_base_env()))

    def make_env_ppo():
        return Monitor(make_base_env())

    # --- build model -------------------------------------------------------
    if algo in ('dqn_vanilla', 'dqn_double', 'dqn_dueling'):
        train_env = DummyVecEnv([make_env_dqn])
        eval_env  = DummyVecEnv([make_env_dqn])
        lr_map    = {'dqn_vanilla': 5e-4, 'dqn_double': 3e-4, 'dqn_dueling': 3e-4}
        net_map   = {
            'dqn_vanilla': [256, 256, 128],
            'dqn_double':  [256, 256, 128],
            'dqn_dueling': [512, 512, 256],
        }
        model = _make_dqn(train_env, lr_map[algo], net_map[algo], log_dir)

    elif algo == 'ppo':
        train_env = DummyVecEnv([make_env_ppo])
        eval_env  = DummyVecEnv([make_env_ppo])
        model     = _make_ppo(train_env, log_dir)

    elif algo == 'sac':
        train_env = DummyVecEnv([make_env_sac])
        eval_env  = DummyVecEnv([make_env_sac])
        model     = _make_sac(train_env, log_dir)

    else:
        raise ValueError(f"Unknown algo: {algo}")

    # --- callbacks ---------------------------------------------------------
    profit_logger = ProfitLogger()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(steps // 30, 1000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    print(f"\n  Training : {run_name}  ({algo.upper()}, {steps:,} steps)")

    model.learn(
        total_timesteps=steps,
        callback=[profit_logger, eval_callback],
        progress_bar=True,
        reset_num_timesteps=True,
        tb_log_name=run_name,
    )

    model.save(f'{model_dir}/final_model')
    np.save(f'{model_dir}/profit_curve.npy', np.array(profit_logger.step_profits))
    print(f"  Saved to {model_dir}/")
    return model, profit_logger


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
ALL_RUNS = [
    # (algo,          run_name,       steps)
    ('dqn_vanilla',  'dqn_vanilla',  300_000),
    ('dqn_double',   'dqn_double',   500_000),
    ('dqn_dueling',  'dqn_dueling',  500_000),
    ('ppo',          'ppo',          500_000),
    ('sac',          'sac',          500_000),
]

ABLATION_RUNS = [
    ('dqn_vanilla', 'ablation_profit_only',  300_000),
    ('ppo',         'ablation_ppo_occ_cap',  300_000),
]


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        default='data/enhanced_data_fixed.csv')
    parser.add_argument('--skip',        nargs='*', default=[],
                        help='Algo names to skip e.g. --skip vanilla double')
    parser.add_argument('--dqn_only',    action='store_true',
                        help='Train DQN variants only (no PPO/SAC)')
    parser.add_argument('--ppo_sac_only',action='store_true',
                        help='Train PPO and SAC only')
    parser.add_argument('--ablation',    action='store_true')
    args = parser.parse_args()

    for d in ['logs', 'models', 'results']:
        os.makedirs(d, exist_ok=True)

    print("\n" + "="*65)
    print("HOTEL DYNAMIC PRICING — FULL ALGORITHM COMPARISON")
    print("="*65)
    print(f"  Data:         {args.data}")
    print(f"  Skip:         {args.skip}")
    print(f"  DQN only:     {args.dqn_only}")
    print(f"  PPO/SAC only: {args.ppo_sac_only}")

    # Filter runs
    runs = ALL_RUNS
    if args.dqn_only:
        runs = [r for r in runs if r[0].startswith('dqn')]
    if args.ppo_sac_only:
        runs = [r for r in runs if r[0] in ('ppo', 'sac')]
    if args.skip:
        runs = [r for r in runs if not any(s in r[0] for s in args.skip)]

    print(f"\n  Running: {[r[1] for r in runs]}\n")

    for algo, name, steps in runs:
        train_one(args.data, algo, name, steps)

    if args.ablation:
        print("\n" + "="*65)
        print("ABLATION RUNS")
        print("="*65)
        for algo, name, steps in ABLATION_RUNS:
            train_one(args.data, algo, name, steps)

    print("\n✅ All training runs complete.")
    print("   Next: python evaluate.py --data", args.data)
