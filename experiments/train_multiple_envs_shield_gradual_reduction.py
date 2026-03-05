"""
Training script for all goal_state environments WITH shield.
The shield protection is gradually reduced by slowly decreasing delta values when performance thresholds are reached.

Trains PPO policies for each environment and saves:
- Trained policy
- Training logs
- Performance plots
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import envs module
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Parse arguments early so SHIELD_MECHANISM is available for module-level constants
_parser = argparse.ArgumentParser(description="Train PPO with gradual shield reduction.")
_parser.add_argument("--mechanism", choices=["delta", "ignore_prob"], default="ignore_prob",
                     help="Shield reduction mechanism (default: ignore_prob)")
_parser.add_argument("--output_dir", required=True,
                     help="Output directory name under experiments/output/")
_args = _parser.parse_args()
SHIELD_MECHANISM = _args.mechanism
OUTPUT_DIR_ARG = _args.output_dir

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure, CSVOutputFormat, HumanOutputFormat
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from minigrid.wrappers import ImgObsWrapper, ReseedWrapper
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import numpy as np
from PIL import Image

from envs.registry import register_env
from shield import DeltaShield
from probabilistic_minigrids import ProbabilisticEnvWrapper
from collections import deque
from feature_extractor import MinigridFeaturesExtractor
from callbacks import GradualShieldReductionCallback
from train_utils import make_video_trigger, save_env_image

# Shield configuration - gradual reduction schedule
INITIAL_DELTA = 0.9  # Start with high protection (used for mechanism='delta')
# Timesteps at which each shield-reduction transition fires
TIMESTEP_SCHEDULE = [1_000_000, 2_000_000, 3_000_000, 4_000_000]
# delta values applied at each corresponding timestep (decreasing protection)
DELTA_SCHEDULE = [0.8, 0.6, 0.4, 0.2]

# ignore_prob schedule (probability of ignoring shield actions) and its fixed delta
IGNORE_PROB_SCHEDULE = [0.2, 0.4, 0.6, 0.8]  # Gradual increase in ignoring probability
IGNORE_PROB_DELTA = 0.9  # Fixed delta value used for DeltaShield when mechanism='ignore_prob'

# Shield mechanism selection
# Shield mechanism selection (set via --mechanism argument, see argparse setup above)
# Options: "delta" or "ignore_prob"

# Define output directory (relative to project root)
OUTPUT_DIR = script_dir / "output" / OUTPUT_DIR_ARG / "shielded_gradual_reduction" / f"{SHIELD_MECHANISM}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# List of goal_state environments to train
GOAL_STATE_ENVS = [
    "CrossingEnv",
    "DistShiftEnv",
    "LavaGapEnv",
]

# Training configuration
TOTAL_TIMESTEPS = 5_000_000  # 5e6
FEATURES_DIM = 128
FIXED_SEED = 42
NUM_ENVS = 24  # Number of parallel environments
BATCH_SIZE = 256  # Batch size for PPO

# Video recording configuration
# Timesteps at which to record a clip (total training timesteps, not vectorized steps)
RECORDING_TIMESTEPS = [
    5_000,        # Very beginning
    500_000,      # Mid first million
    950_000,      # Towards end of first million
    1_000_000,    # 1M
    2_000_000,    # 2M
    3_000_000,    # 3M
    4_000_000,    # 4M
    5_000_000,    # End
]
VIDEO_LENGTH = 200  # Max frames per clip










def plot_training_results(log_dir: Path, env_name: str, output_path: Path, shield_disable_timestep=None):
    """Load training results and create performance plots."""
    try:
        # Read shield_gradual_reduction CSV from PPO logger
        csv_filename = f"shield_gradual_reduction_{SHIELD_MECHANISM}.csv"
        progress_file = log_dir / csv_filename
        df = pd.read_csv(progress_file)
        
        # Create performance plot (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Episode reward mean over timesteps
        axes[0, 0].plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], linewidth=2)
        if shield_disable_timestep:
            axes[0, 0].axvline(x=shield_disable_timestep, color='red', linestyle='--', alpha=0.7, 
                            label='Final Shield Transition')
            axes[0, 0].legend()
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Episode Reward (mean)')
        axes[0, 0].set_title(f'{env_name}: Training Rewards Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode length mean over timesteps
        axes[0, 1].plot(df['time/total_timesteps'], df['rollout/ep_len_mean'], linewidth=2)
        if shield_disable_timestep:
            axes[0, 1].axvline(x=shield_disable_timestep, color='red', linestyle='--', alpha=0.7, 
                            label='Final Shield Transition')
            axes[0, 1].legend()
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Episode Length (mean)')
        axes[0, 1].set_title(f'{env_name}: Episode Lengths Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: KL divergence over timesteps
        if 'train/approx_kl' in df.columns:
            axes[1, 0].plot(df['time/total_timesteps'], df['train/approx_kl'], linewidth=2, color='orange')
            if shield_disable_timestep:
                axes[1, 0].axvline(x=shield_disable_timestep, color='red', linestyle='--', alpha=0.7, 
                                label='Final Shield Transition')
                axes[1, 0].legend()
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Approx KL Divergence')
            axes[1, 0].set_title(f'{env_name}: KL Divergence Over Time')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'KL Divergence data not available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(f'{env_name}: KL Divergence Over Time')
        
        # Plot 4: Entropy over timesteps
        if 'train/entropy_loss' in df.columns:
            axes[1, 1].plot(df['time/total_timesteps'], df['train/entropy_loss'], linewidth=2, color='green')
            if shield_disable_timestep:
                axes[1, 1].axvline(x=shield_disable_timestep, color='red', linestyle='--', alpha=0.7, 
                                label='Final Shield Transition')
                axes[1, 1].legend()
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('Entropy Loss')
            axes[1, 1].set_title(f'{env_name}: Entropy Loss Over Time')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Entropy data not available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f'{env_name}: Entropy Loss Over Time')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create separate shield plot if shield data exists
        shield_output_path = output_path.parent / f"{env_name}_shield_progression.png"
        has_shield_data = ('shield/stage' in df.columns 
                          or 'shield/delta' in df.columns 
                          or 'shield/ignore_prob' in df.columns)
        
        if has_shield_data:
            fig_shield, axes_shield = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Shield stage progression
            if 'shield/stage' in df.columns:
                axes_shield[0].plot(df['time/total_timesteps'], df['shield/stage'], 
                                  linewidth=2, color='orange', marker='o', markersize=3)
                axes_shield[0].set_xlabel('Timesteps')
                axes_shield[0].set_ylabel('Shield Stage')
                axes_shield[0].set_title(f'{env_name}: Shield Stage Progression')
                axes_shield[0].grid(True, alpha=0.3)
                axes_shield[0].set_ylim(-0.1, max(df['shield/stage']) + 0.5)
            else:
                axes_shield[0].text(0.5, 0.5, 'Shield stage data not available', 
                              ha='center', va='center', transform=axes_shield[0].transAxes)
                axes_shield[0].set_title(f'{env_name}: Shield Stage Progression')
            
            # Plot 2: Shield parameter progression (delta or ignore_prob)
            if 'shield/delta' in df.columns:
                axes_shield[1].plot(df['time/total_timesteps'], df['shield/delta'], 
                                  linewidth=2, color='green', marker='o', markersize=3)
                axes_shield[1].set_xlabel('Timesteps')
                axes_shield[1].set_ylabel('Shield Delta (δ)')
                axes_shield[1].set_title(f'{env_name}: Shield Protection Level Over Time')
                axes_shield[1].grid(True, alpha=0.3)
                axes_shield[1].set_ylim(-0.05, 1.0)
            
            elif 'shield/ignore_prob' in df.columns:
                axes_shield[1].plot(df['time/total_timesteps'], df['shield/ignore_prob'], 
                                  linewidth=2, color='blue', marker='o', markersize=3)
                axes_shield[1].set_xlabel('Timesteps')
                axes_shield[1].set_ylabel('Shield Ignore Probability')
                axes_shield[1].set_title(f'{env_name}: Shield Ignore Probability Over Time')
                axes_shield[1].grid(True, alpha=0.3)
                axes_shield[1].set_ylim(-0.05, 1.05)
            
            else:
                axes_shield[1].text(0.5, 0.5, 'Shield parameter data not available', 
                              ha='center', va='center', transform=axes_shield[1].transAxes)
                axes_shield[1].set_title(f'{env_name}: Shield Parameter Progression')
            
            plt.tight_layout()
            plt.savefig(shield_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Shield progression plot saved to: {shield_output_path}")
        
        # Print summary statistics
        print(f"\n  Training Summary:")
        print(f"    Total timesteps: {df['time/total_timesteps'].iloc[-1]:.0f}")
        print(f"    Final episode reward (mean): {df['rollout/ep_rew_mean'].iloc[-1]:.2f}")
        print(f"    Final episode length (mean): {df['rollout/ep_len_mean'].iloc[-1]:.2f}")
        print(f"    Best episode reward (mean): {df['rollout/ep_rew_mean'].max():.2f}")
        
        if 'shield/stage' in df.columns:
            print(f"    Final shield stage: {df['shield/stage'].iloc[-1]:.0f}")
        if 'shield/delta' in df.columns:
            print(f"    Final shield delta: {df['shield/delta'].iloc[-1]:.1f}")
        if 'shield/ignore_prob' in df.columns:
            print(f"    Final shield ignore prob: {df['shield/ignore_prob'].iloc[-1]:.1f}")
        
    except Exception as e:
        print(f"  Error creating plot: {e}")
        import traceback
        traceback.print_exc()





def train_environment(env_name: str):
    """Train a PPO policy for a single environment with gradual shield reduction."""
    print(f"\n{'='*80}")
    print(f"Training {env_name} with GRADUAL SHIELD REDUCTION")
    print(f"{'='*80}")
    
    # Create environment-specific directories
    env_dir = OUTPUT_DIR / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    
    policy_path = env_dir / f"PPO_{env_name}"
    log_dir = env_dir / "ppo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = env_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    plot_path = env_dir / f"{env_name}_training_plot.png"
    env_image_path = env_dir / f"{env_name}_environment.png"
    
    # Register and create environments
    print(f"\n1. Setting up {NUM_ENVS} environments...")
    register_env(f"./envs/configs/goal_state/{env_name}.yaml")
    
    def make_env():
        env = gym.make(f"{env_name}-v0")
        env = ImgObsWrapper(env)
        env = ReseedWrapper(env, seeds=[FIXED_SEED])
        env.unwrapped.add_lava()

        # Build Storm model on the wrapper, then attach shield.
        # The initial delta depends on the mechanism:
        #   - 'delta': delta will be varied by the callback, so start at INITIAL_DELTA.
        #   - 'ignore_prob': delta stays fixed at IGNORE_PROB_DELTA; only ignore_prob varies.
        initial_delta = INITIAL_DELTA if SHIELD_MECHANISM == "delta" else IGNORE_PROB_DELTA
        env.reset()
        model, _ = env.unwrapped.convert_to_probabilistic_storm()
        shield = DeltaShield(model, "Pmin=? [F \"lava\"]", delta=initial_delta)
        env.unwrapped.set_shield(shield)

        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    
    # Add monitoring to track episode statistics for logging
    env = VecMonitor(env, filename=str(log_dir / "monitor"))

    # Add video recording at configured timesteps
    env = VecVideoRecorder(
        env,
        video_folder=str(video_dir),
        record_video_trigger=make_video_trigger(RECORDING_TIMESTEPS, NUM_ENVS),
        video_length=VIDEO_LENGTH,
        name_prefix=env_name,
    )
    
    _displayed_delta = INITIAL_DELTA if SHIELD_MECHANISM == "delta" else IGNORE_PROB_DELTA
    print(f"   {NUM_ENVS} environments created with shield (δ={_displayed_delta})")
    
    # Save environment image (using first environment from the vectorized env)
    print(f"\n2. Saving environment image...")
    save_env_image(env.get_attr('unwrapped')[0], env_name, env_image_path)
    
    # Setup policy
    print(f"\n3. Configuring PPO policy...")
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )
    
    # Configure logging with descriptive CSV filename
    csv_filename = f"shield_gradual_reduction_{SHIELD_MECHANISM}.csv"
    csv_logger = CSVOutputFormat(str(log_dir / csv_filename))
    human_logger = HumanOutputFormat(sys.stdout)
    ppo_logger = configure(folder=None, format_strings=[])
    ppo_logger.output_formats = [human_logger, csv_logger]
    
    # Create PPO model
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        batch_size=BATCH_SIZE
    )
    model.set_logger(ppo_logger)
    
    print(f"   PPO model initialized")
    
    # Create callback for gradual shield reduction
    if SHIELD_MECHANISM == "delta":
        shield_callback = GradualShieldReductionCallback(
            mechanism="delta",
            delta_schedule=DELTA_SCHEDULE,
            timestep_schedule=TIMESTEP_SCHEDULE,
            verbose=1,
        )
    else:  # ignore_prob
        shield_callback = GradualShieldReductionCallback(
            mechanism="ignore_prob",
            ignore_prob_schedule=IGNORE_PROB_SCHEDULE,
            ignore_prob_delta=IGNORE_PROB_DELTA,
            timestep_schedule=TIMESTEP_SCHEDULE,
            verbose=1,
        )
    
    # Train
    print(f"\n4. Training for {TOTAL_TIMESTEPS:,.0f} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=shield_callback)
    
    # Save policy
    print(f"\n5. Saving trained policy...")
    model.save(str(policy_path))
    print(f"   Policy saved to: {policy_path}")
    
    # Create performance plot
    print(f"\n6. Creating performance plot...")
    final_transition = shield_callback.stage_transitions[-1]['timestep'] if shield_callback.stage_transitions else None
    plot_training_results(
        log_dir, 
        env_name, 
        plot_path,
        shield_disable_timestep=final_transition
    )
    print(f"   Plot saved to: {plot_path}")
    
    # Cleanup
    env.close()
    
    print(f"\n✓ Completed training for {env_name}")
    return shield_callback.stage_transitions


def main():
    """Train all goal_state environments with gradual shield reduction."""
    print("="*80)
    print("TRAINING ALL GOAL_STATE ENVIRONMENTS (WITH GRADUAL SHIELD REDUCTION)")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Number of environments: {len(GOAL_STATE_ENVS)}")
    print(f"Environments found: {', '.join(GOAL_STATE_ENVS)}")
    print(f"Timesteps per environment: {TOTAL_TIMESTEPS:,.0f}")
    print(f"Fixed seed: {FIXED_SEED}")
    print(f"Parallel environments per training: {NUM_ENVS}")
    print(f"Initial delta: {INITIAL_DELTA}")
    print(f"Shield mechanism: {SHIELD_MECHANISM}")
    
    successful = []
    failed = []
    all_transitions = {}
    
    for i, env_name in enumerate(GOAL_STATE_ENVS, 1):
        print(f"\n\nProgress: {i}/{len(GOAL_STATE_ENVS)}")
        try:
            transitions = train_environment(env_name)
            successful.append(env_name)
            all_transitions[env_name] = transitions
        except Exception as e:
            print(f"\n✗ FAILED to train {env_name}: {e}")
            failed.append(env_name)
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\n✓ Successfully trained: {len(successful)}/{len(GOAL_STATE_ENVS)}")
    for env_name in successful:
        print(f"  - {env_name}")
        if env_name in all_transitions and all_transitions[env_name]:
            final_stage = all_transitions[env_name][-1]
            if SHIELD_MECHANISM == "delta":
                final_value = final_stage.get('delta', 'N/A')
                print(f"    Final shield state: δ={final_value:.1f} at timestep {final_stage['timestep']}")
            else:  # ignore_prob
                final_value = final_stage.get('ignore_prob', 'N/A')
                print(f"    Final shield state: ignore_prob={final_value:.1f} at timestep {final_stage['timestep']}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(GOAL_STATE_ENVS)}")
        for env_name in failed:
            print(f"  - {env_name}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()


