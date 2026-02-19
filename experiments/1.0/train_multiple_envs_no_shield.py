"""
Training script for all goal_state environments WITHOUT shield.
Trains PPO policies for each environment and saves:
- Trained policy
- Training logs
- Performance plots
"""

import sys
from pathlib import Path

# Add parent directory to path to import envs module
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from minigrid.wrappers import ImgObsWrapper, ReseedWrapper
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import numpy as np
from PIL import Image

from envs.registry import register_env

# Define output directory (relative to project root)
OUTPUT_DIR = project_root / "experiments" / "1.0" / "unshielded3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# List of goal_state environments to train
GOAL_STATE_ENVS = [
    "CrossingEnv",
    "DistShiftEnv",
    "FourRoomsEnv",
    "LavaGapEnv",
    "LockedRoomEnv",
    "MultiRoomEnv",
]

# Training configuration
TOTAL_TIMESTEPS = 5_000_000  # 5e6
FEATURES_DIM = 128
FIXED_SEED = 42
NUM_ENVS = 24  # Number of parallel environments
BATCH_SIZE = 256  # Batch size for PPO


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Custom CNN feature extractor for MiniGrid environments."""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def plot_training_results(log_dir: Path, env_name: str, output_path: Path):
    """Load training results and create performance plots."""
    try:
        # Read progress.csv from PPO logger (same approach as notebook)
        progress_file = log_dir / "progress.csv"
        df = pd.read_csv(progress_file)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Episode reward mean over timesteps
        axes[0, 0].plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], linewidth=2)
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Episode Reward (mean)')
        axes[0, 0].set_title(f'{env_name}: Training Rewards Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode length mean over timesteps
        axes[0, 1].plot(df['time/total_timesteps'], df['rollout/ep_len_mean'], linewidth=2)
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Episode Length (mean)')
        axes[0, 1].set_title(f'{env_name}: Episode Lengths Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: KL divergence over timesteps
        if 'train/approx_kl' in df.columns:
            axes[1, 0].plot(df['time/total_timesteps'], df['train/approx_kl'], linewidth=2, color='orange')
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
        
        # Print summary statistics
        print(f"\n  Training Summary:")
        print(f"    Total timesteps: {df['time/total_timesteps'].iloc[-1]:.0f}")
        print(f"    Final episode reward (mean): {df['rollout/ep_rew_mean'].iloc[-1]:.2f}")
        print(f"    Final episode length (mean): {df['rollout/ep_len_mean'].iloc[-1]:.2f}")
        print(f"    Best episode reward (mean): {df['rollout/ep_rew_mean'].max():.2f}")
        if 'train/approx_kl' in df.columns:
            print(f"    Final KL divergence: {df['train/approx_kl'].iloc[-1]:.6f}")
        if 'train/entropy_loss' in df.columns:
            print(f"    Final entropy loss: {df['train/entropy_loss'].iloc[-1]:.6f}")
        
    except Exception as e:
        print(f"  Error creating plot: {e}")
        import traceback
        traceback.print_exc()


def save_env_image(env, env_name: str, output_path: Path):
    """Render and save an image of the environment."""
    try:
        # Reset to get initial state
        env.reset()
        # Access the unwrapped environment to render properly
        # The ReseedWrapper and ImgObsWrapper don't have render, so go to base env
        base_env = env.env
        rgb_array = base_env.render()
        if rgb_array is not None:
            # Save as image
            img = Image.fromarray(rgb_array)
            img.save(output_path)
            print(f"   Environment image saved to: {output_path}")
        else:
            print(f"   Warning: Could not render environment")
    except Exception as e:
        print(f"   Error saving environment image: {e}")
        import traceback
        traceback.print_exc()


def train_environment(env_name: str):
    """Train a PPO policy for a single environment."""
    print(f"\n{'='*80}")
    print(f"Training {env_name}")
    print(f"{'='*80}")
    
    # Create environment-specific directories
    env_dir = OUTPUT_DIR / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    
    policy_path = env_dir / f"PPO_{env_name}"
    log_dir = env_dir / "ppo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
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
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    
    # Add monitoring to track episode statistics for logging
    env = VecMonitor(env, filename=str(log_dir / "monitor"))
    
    print(f"   {NUM_ENVS} environments created and wrapped in DummyVecEnv with VecMonitor")
    
    # Save environment image (using first environment from the vectorized env)
    print(f"\n2. Saving environment image...")
    save_env_image(env.get_attr('unwrapped')[0], env_name, env_image_path)
    
    # Setup policy
    print(f"\n3. Configuring PPO policy...")
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )
    
    # Configure logging
    ppo_logger = configure(str(log_dir), ["stdout", "csv"])
    
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
    
    # Train
    print(f"\n4. Training for {TOTAL_TIMESTEPS:,.0f} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save policy
    print(f"\n5. Saving trained policy...")
    model.save(str(policy_path))
    print(f"   Policy saved to: {policy_path}")
    
    # Create performance plot
    print(f"\n6. Creating performance plot...")
    plot_training_results(log_dir, env_name, plot_path)
    print(f"   Plot saved to: {plot_path}")
    
    # Cleanup
    env.close()
    
    print(f"\n✓ Completed training for {env_name}")


def main():
    """Train all goal_state environments."""
    print("="*80)
    print("TRAINING ALL GOAL_STATE ENVIRONMENTS (WITHOUT SHIELD)")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Number of environments: {len(GOAL_STATE_ENVS)}")
    print(f"Environments found: {', '.join(GOAL_STATE_ENVS)}")
    print(f"Timesteps per environment: {TOTAL_TIMESTEPS:,.0f}")
    print(f"Fixed seed: {FIXED_SEED}")
    
    successful = []
    failed = []
    
    for i, env_name in enumerate(GOAL_STATE_ENVS, 1):
        print(f"\n\nProgress: {i}/{len(GOAL_STATE_ENVS)}")
        try:
            train_environment(env_name)
            successful.append(env_name)
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
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(GOAL_STATE_ENVS)}")
        for env_name in failed:
            print(f"  - {env_name}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
