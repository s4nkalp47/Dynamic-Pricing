"""
evaluate.py — Full evaluation pipeline

Loads all trained models (DQN variants + PPO + SAC) + baselines.
All agents evaluated on identical decision sequences.

Usage:
    python evaluate.py --data data/enhanced_data_fixed.csv
    python evaluate.py --data data/enhanced_data_fixed.csv --episodes 20 --robustness
"""

import os
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, SAC

from src.envs.hotel_env import HotelPricingEnv
from src.agents.baselines import TraditionalOptimizer, FixedMatch, FixedUndercut
from src.evaluation.evaluator import evaluate_all, results_to_dataframe
from src.evaluation.stats import pairwise_significance, robustness_table
from src.evaluation.plots import (
    fig1_performance_comparison,
    fig2_demand_stratified,
    fig3_learning_curves,
    fig4_robustness_heatmap,
)


# ---------------------------------------------------------------------------
# Model loader — handles DQN / PPO / SAC transparently
# ---------------------------------------------------------------------------
MODEL_CLASSES = {
    'dqn': DQN,
    'ppo': PPO,
    'sac': SAC,
}

def _load_model(run_name: str):
    """Load best_model if available, else final_model. Returns None if missing."""
    algo = 'ppo' if 'ppo' in run_name else ('sac' if 'sac' in run_name else 'dqn')
    cls  = MODEL_CLASSES[algo]

    for fname in ['best_model', 'final_model']:
        for ext in ['', '.zip']:
            path = f'models/{run_name}/{fname}{ext}'
            if os.path.exists(path):
                print(f"  Loading {run_name} from {path}")
                return cls.load(path)

    print(f"  [WARN] No model found at models/{run_name}/ — skipping.")
    return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_main(data_path: str, n_episodes: int):
    print("\n" + "="*65)
    print("MAIN EVALUATION — ALL ALGORITHMS")
    print("="*65)

    env = HotelPricingEnv(data_path, reward_variant='profit_only', seed=0)

    # RL agents
    agents = {}
    for display_name, run_name in [
        ('DQN',         'dqn_vanilla'),
        ('Double DQN',  'dqn_double'),
        ('Dueling DQN', 'dqn_dueling'),
        ('PPO',         'ppo'),
        ('SAC',         'sac'),
    ]:
        m = _load_model(run_name)
        if m is not None:
            agents[display_name] = m

    # Baselines
    agents['Traditional Opt.'] = TraditionalOptimizer(env)
    agents['Fixed Match']       = FixedMatch()
    agents['Fixed Undercut']    = FixedUndercut(0.05)

    print(f"\n  Agents loaded: {list(agents.keys())}")

    results = evaluate_all(env, agents, n_episodes=n_episodes)

    # Tables
    df = results_to_dataframe(results, reference='Traditional Opt.')
    df.to_csv('results/summary_table.csv', index=False)
    print("\n📋 Summary Table:")
    print(df.to_string(index=False))

    sig_df = pairwise_significance(results, reference='Traditional Opt.')
    sig_df.to_csv('results/significance_table.csv', index=False)
    print("\n📋 Significance Tests (vs Traditional Opt.):")
    print(sig_df.to_string(index=False))

    # Figures
    os.makedirs('results', exist_ok=True)
    print("\n📊 Generating figures...")
    fig1_performance_comparison(results)
    fig2_demand_stratified(
        results,
        highlight=['DQN', 'PPO', 'SAC', 'Traditional Opt.']
    )

    # Learning curves
    curves = {}
    for name, run in [('DQN', 'dqn_vanilla'), ('Double DQN', 'dqn_double'),
                      ('Dueling DQN', 'dqn_dueling'), ('PPO', 'ppo'), ('SAC', 'sac')]:
        p = f'models/{run}/profit_curve.npy'
        if os.path.exists(p):
            curves[name] = np.load(p)
    if curves:
        fig3_learning_curves(curves)

    return results


# ---------------------------------------------------------------------------
# Robustness sweep
# ---------------------------------------------------------------------------
def evaluate_robustness(data_path: str, n_episodes: int):
    print("\n" + "="*65)
    print("ROBUSTNESS — varying elasticity at test time")
    print("="*65)

    results_by_eps = {}

    for eps in [-0.5, -0.6, -0.8, -1.0, -1.2]:
        print(f"\n  ε = {eps}")
        env = HotelPricingEnv(data_path, reward_variant='profit_only', seed=0)
        env.elasticity = eps

        agents = {}
        for display_name, run_name in [
            ('DQN', 'dqn_vanilla'), ('PPO', 'ppo'), ('SAC', 'sac')
        ]:
            m = _load_model(run_name)
            if m is not None:
                agents[display_name] = m
        agents['Traditional Opt.'] = TraditionalOptimizer(env)

        results_by_eps[eps] = evaluate_all(env, agents, n_episodes=max(n_episodes // 2, 5))

    rob_df = robustness_table(results_by_eps)
    rob_df.to_csv('results/robustness_table.csv', index=False)
    print("\n📋 Robustness Table:")
    print(rob_df.to_string(index=False))
    fig4_robustness_heatmap(rob_df)

    return results_by_eps


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='data/enhanced_data_fixed.csv')
    parser.add_argument('--episodes',   type=int, default=20)
    parser.add_argument('--robustness', action='store_true')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    main_results = evaluate_main(args.data, args.episodes)

    if args.robustness:
        evaluate_robustness(args.data, args.episodes)

    print("\n✅ Evaluation complete. Outputs in results/")


# ---------------------------------------------------------------------------
# 8D STATE EVALUATION
# ---------------------------------------------------------------------------
def evaluate_8d(data_path: str, n_episodes: int, reference_profit: float):
    print("\n" + "="*65)
    print("8D STATE EXPERIMENT EVALUATION")
    print("="*65)

    env_8d = HotelPricingEnv(data_path, reward_variant='profit_only',
                             state_variant='8d', seed=0)

    agents = {}
    for display_name, run_name in [('DQN 8D', 'dqn_8d'), ('PPO 8D', 'ppo_8d')]:
        m = _load_model(run_name)
        if m is not None:
            agents[display_name] = m

    if not agents:
        print("  [SKIP] No 8D models found. Run: python train.py --experiment_8d")
        return

    # Add baselines for comparison
    agents['Traditional Opt.'] = TraditionalOptimizer(env_8d)

    results = evaluate_all(env_8d, agents, n_episodes=n_episodes)
    df      = results_to_dataframe(results, reference='Traditional Opt.')
    df.to_csv('results/summary_8d.csv', index=False)

    print("\n📋 8D State Results vs 5D:")
    print(f"  {'Method':<15} {'Mean Profit':>12} {'vs Oracle':>10} {'vs 5D counterpart':>18}")
    print("  " + "-"*55)

    five_d = {'DQN 8D': reference_profit}  # placeholder, filled below
    for _, row in df.iterrows():
        print(f"  {row['Method']:<15} {row['Mean Profit']:>12,.0f} "
              f"{row['vs Reference (%)']:>9.1f}%")

    print(f"\n  Reference profit (5D baseline): {reference_profit:.0f} INR")
