"""
Training script for all goal_state environments WITHOUT shield using FULL observability.
Unlike the partial observation version, this uses FullyObsWrapper to give the agent
the complete grid view and uses a dynamic CNN architecture that adapts to different 
environment sizes.

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
from minigrid.wrappers import ImgObsWrapper, ReseedWrapper, FullyObsWrapper
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import numpy as np
from PIL import Image

from envs.registry import register_env

# Define output directory (relative to project root)
OUTPUT_DIR = project_root / "experiments" / "1.0" / "fully_observable"
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
TOTAL_TIMESTEPS = 200000  # 2e5
FEATURES_DIM = 128
FIXED_SEED = 42


class DynamicMinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dynamic CNN feature extractor for MiniGrid environments with full observability.
    Uses adaptive pooling to handle different grid sizes (e.g., 8x8, 19x19).
    This ensures consistent output dimensions regardless of environment size.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # CNN layers - deeper network for larger full observations
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Adaptive pooling to fixed size regardless of input dimensions
        # This ensures consistent output for different environment sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size: 128 channels * 4 * 4 = 2048
        n_flatten = 128 * 4 * 4
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)


def plot_training_results(log_dir: Path, env_name: str, output_path: Path):
    """Load training results and create performance plots."""
    try:
        # Read progress.csv from PPO logger (same approach as notebook)
        progress_file = log_dir / "progress.csv"
        df = pd.read_csv(progress_file)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Episode reward mean over timesteps
        axes[0].plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], linewidth=2)
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Episode Reward (mean)')
        axes[0].set_title(f'{env_name}: Training Rewards Over Time (Full Obs)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Episode length mean over timesteps
        axes[1].plot(df['time/total_timesteps'], df['rollout/ep_len_mean'], linewidth=2)
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Episode Length (mean)')
        axes[1].set_title(f'{env_name}: Episode Lengths Over Time (Full Obs)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\n  Training Summary:")
        print(f"    Total timesteps: {df['time/total_timesteps'].iloc[-1]:.0f}")
        print(f"    Final episode reward (mean): {df['rollout/ep_rew_mean'].iloc[-1]:.2f}")
        print(f"    Final episode length (mean): {df['rollout/ep_len_mean'].iloc[-1]:.2f}")
        print(f"    Best episode reward (mean): {df['rollout/ep_rew_mean'].max():.2f}")
        
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
        base_env = env.unwrapped.env
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
    print(f"Training {env_name} (FULL OBSERVABILITY)")
    print(f"{'='*80}")
    
    # Create environment-specific directories
    env_dir = OUTPUT_DIR / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    
    policy_path = env_dir / f"PPO_{env_name}_fullobs"
    log_dir = env_dir / "ppo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_path = env_dir / f"{env_name}_training_plot_fullobs.png"
    env_image_path = env_dir / f"{env_name}_environment.png"
    
    # Register and create environment
    print(f"\n1. Setting up environment...")
    register_env(f"./envs/configs/goal_state/{env_name}.yaml")
    
    env = gym.make(f"{env_name}-v0")
    env = FullyObsWrapper(env)  # Add full observability BEFORE ImgObsWrapper
    env = ImgObsWrapper(env)
    env = ReseedWrapper(env, seeds=[FIXED_SEED])
    env.unwrapped.add_lava()
    
    # Print observation space info
    obs_shape = env.observation_space.shape
    print(f"   Environment created and wrapped")
    print(f"   Observation space shape: {obs_shape}")
    
    # Save environment image
    print(f"\n2. Saving environment image...")
    save_env_image(env, env_name, env_image_path)
    
    # Setup policy
    print(f"\n3. Configuring PPO policy...")
    policy_kwargs = dict(
        features_extractor_class=DynamicMinigridFeaturesExtractor,
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
    )
    model.set_logger(ppo_logger)
    
    print(f"   PPO model initialized with dynamic CNN")
    
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
    """Train all goal_state environments with full observability."""
    print("="*80)
    print("TRAINING ALL GOAL_STATE ENVIRONMENTS (FULL OBSERVABILITY - NO SHIELD)")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Number of environments: {len(GOAL_STATE_ENVS)}")
    print(f"Environments: {', '.join(GOAL_STATE_ENVS)}")
    print(f"Timesteps per environment: {TOTAL_TIMESTEPS:,.0f}")
    print(f"Fixed seed: {FIXED_SEED}")
    print(f"\nUsing FullyObsWrapper - agent sees entire grid")
    print(f"Using dynamic CNN with adaptive pooling for variable grid sizes")
    
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
