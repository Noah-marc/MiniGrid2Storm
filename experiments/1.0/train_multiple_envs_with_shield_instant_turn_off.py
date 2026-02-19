"""
Training script for all goal_state environments WITH shield.
The shield is disabled automatically when a reward threshold is reached.

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

# Define output directory (relative to project root)
OUTPUT_DIR = project_root / "experiments" / "1.0" / "shielded_instant_turn_off"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# List of goal_state environments to train
GOAL_STATE_ENVS = [
    "CrossingEnv",
    "DistShiftEnv",
    # "FourRoomsEnv",
    "LavaGapEnv",
    # "LockedRoomEnv",
    # "MultiRoomEnv",
]

# Training configuration
TOTAL_TIMESTEPS = 200000  # 2e5
FEATURES_DIM = 128
FIXED_SEED = 42

# Shield configuration
SHIELD_DELTA = 0.5  # Delta parameter for DeltaShield
REWARD_THRESHOLD = 0.6  # Disable shield when mean reward reaches this value


class ShieldDisablerCallback(BaseCallback):
    """
    Custom callback that monitors training progress and disables the shield
    when the mean episode reward reaches a specified threshold.
    """
    
    def __init__(self, env, reward_threshold: float, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.reward_threshold = reward_threshold
        self.shield_disabled = False
        self.disable_timestep = None
    
    def _on_step(self) -> bool:
        """
        Called after every step. We check if we should disable the shield.
        Returns True to continue training, False to stop.
        """
        # Only check every N steps to avoid overhead (check every rollout)
        if self.n_calls % 2048 == 0:  # Default PPO n_steps is 2048
            # Get the mean reward from the logger
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                
                # Check if we should disable the shield
                if not self.shield_disabled and mean_reward >= self.reward_threshold:
                    self.shield_disabled = True
                    self.disable_timestep = self.num_timesteps
                    
                    # Disable the shield by accessing the unwrapped environment
                    self.env.unwrapped.remove_shield()
                    
                    if self.verbose > 0:
                        print(f"\n{'='*80}")
                        print(f"🎯 SHIELD DISABLED at timestep {self.num_timesteps}")
                        print(f"   Mean reward reached threshold: {mean_reward:.3f} >= {self.reward_threshold}")
                        print(f"   Continuing training without shield...")
                        print(f"{'='*80}\n")
        
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.shield_disabled:
            print(f"\n   Shield was disabled at timestep {self.disable_timestep}")
        else:
            print(f"\n   Shield remained active throughout training (threshold not reached)")

class ShieldHardCutoffCallback(BaseCallback):
    """
    Hard cutoff shielding callback. Removes the sield once for 20 episodes the mean reward is above the threshold
    - Tracks rolling mean episode reward
    - Defines an empirical reference reward R_ref
    - Disables the shield once reward >= alpha * R_ref
    """

    def __init__(
        self,
        nr_episodes: int = 10,
        threshold: float = 0.95,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.nr_episodes = nr_episodes
        self.threshold = threshold

        self.ep_rewards = deque(maxlen=self.nr_episodes)
        self.shield_active = True
        self.cutoff_timestep = None

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        """
        if self.shield_active:
            infos = self.locals.get("infos", [])

            for info in infos:
                # Monitor wrapper adds this when an episode ends
                if "episode" in info:
                    ep_rew = info["episode"]["r"]
                    self.ep_rewards.append(ep_rew)

                    # Only act once we have a full window
                    if len(self.ep_rewards) == self.nr_episodes:
                        mean_rew = np.mean(self.ep_rewards)

                        # Update empirical reference reward
                        if mean_rew > self.threshold:
                            # Access the actual environment through the vectorized wrapper
                            # self.training_env is a DummyVecEnv, need to get env 0
                            env = self.training_env.envs[0]
                            # Unwrap through wrappers until we reach ProbabilisticEnvWrapper
                            while hasattr(env, 'env') and not isinstance(env, ProbabilisticEnvWrapper):
                                env = env.env
                            # Now we have the ProbabilisticEnvWrapper
                            if isinstance(env, ProbabilisticEnvWrapper):
                                env.remove_shield()
                            else:
                                raise RuntimeError("Could not find ProbabilisticEnvWrapper to remove shield")
                            
                            self.shield_active = False
                            self.cutoff_timestep = self.num_timesteps
                            self.logger.record("shield_cutoff_timestep", self.num_timesteps)
                            
                            if self.verbose > 0:
                                print(f"\n{'='*80}")
                                print(f"🎯 SHIELD DISABLED at timestep {self.num_timesteps}")
                                print(f"   Mean reward over {self.nr_episodes} episodes: {mean_rew:.3f} >= {self.threshold}")
                                print(f"   Continuing training without shield...")
                                print(f"{'='*80}\n")
        return True

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


def plot_training_results(log_dir: Path, env_name: str, output_path: Path, shield_disable_timestep: int = None):
    """Load training results and create performance plots."""
    try:
        # Read progress.csv from PPO logger (same approach as notebook)
        progress_file = log_dir / "progress.csv"
        df = pd.read_csv(progress_file)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Episode reward mean over timesteps
        axes[0].plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], linewidth=2)
        
        # Add vertical line where shield was disabled
        if shield_disable_timestep is not None:
            axes[0].axvline(x=shield_disable_timestep, color='red', linestyle='--', 
                          linewidth=2, label=f'Shield disabled')
            axes[0].legend()
        
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Episode Reward (mean)')
        axes[0].set_title(f'{env_name}: Training Rewards Over Time (With Shield)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Episode length mean over timesteps
        axes[1].plot(df['time/total_timesteps'], df['rollout/ep_len_mean'], linewidth=2)
        
        if shield_disable_timestep is not None:
            axes[1].axvline(x=shield_disable_timestep, color='red', linestyle='--', 
                          linewidth=2, label=f'Shield disabled')
            axes[1].legend()
        
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Episode Length (mean)')
        axes[1].set_title(f'{env_name}: Episode Lengths Over Time (With Shield)')
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
        if shield_disable_timestep is not None:
            print(f"    Shield disabled at timestep: {shield_disable_timestep}")
        
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
    print(f"Training {env_name} (WITH SHIELD)")
    print(f"{'='*80}")
    
    # Create environment-specific directories
    env_dir = OUTPUT_DIR / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    
    policy_path = env_dir / f"PPO_{env_name}_shielded"
    log_dir = env_dir / "ppo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_path = env_dir / f"{env_name}_training_plot_shielded.png"
    env_image_path = env_dir / f"{env_name}_environment.png"
    
    # Register and create environment
    print(f"\n1. Setting up environment...")
    register_env(f"./envs/configs/goal_state/{env_name}.yaml")
    
    env = gym.make(f"{env_name}-v0")
    env = ImgObsWrapper(env)
    env = ReseedWrapper(env, seeds=[FIXED_SEED])
    env.unwrapped.add_lava()
    
    print(f"   Environment created and wrapped")
    
    # Setup shield
    print(f"\n2. Setting up DeltaShield...")
    env.reset()
    model, _ = env.unwrapped.convert_to_probabilistic_storm()
    shield = DeltaShield(model, "Pmin=? [F \"lava\"]", delta=SHIELD_DELTA)
    env.unwrapped.set_shield(shield)
    print(f"   Shield configured with delta={SHIELD_DELTA}")
    print(f"   Shield will be disabled when mean reward >= {REWARD_THRESHOLD}")
    
    # Save environment image
    print(f"\n3. Saving environment image...")
    save_env_image(env, env_name, env_image_path)
    
    # Setup policy
    print(f"\n4. Configuring PPO policy...")
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
    )
    model.set_logger(ppo_logger)
    
    print(f"   PPO model initialized")
    
    # Create callback to disable shield at threshold
    shield_callback = ShieldHardCutoffCallback(
        nr_episodes=100, 
        threshold=REWARD_THRESHOLD,
        verbose=1
    )
    
    # Train
    print(f"\n5. Training for {TOTAL_TIMESTEPS:,.0f} timesteps...")
    print(f"   Monitoring for {shield_callback.nr_episodes} episodes mean reward threshold to disable shield...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=shield_callback)
    
    # Save policy
    print(f"\n6. Saving trained policy...")
    model.save(str(policy_path))
    print(f"   Policy saved to: {policy_path}")
    
    # Create performance plot
    print(f"\n7. Creating performance plot...")
    plot_training_results(
        log_dir, 
        env_name, 
        plot_path,
        shield_disable_timestep=shield_callback.cutoff_timestep
    )
    print(f"   Plot saved to: {plot_path}")
    
    # Cleanup
    env.close()
    
    print(f"\n✓ Completed training for {env_name}")
    
    return shield_callback.cutoff_timestep


def main():
    """Train all goal_state environments with shield."""
    print("="*80)
    print("TRAINING ALL GOAL_STATE ENVIRONMENTS (WITH SHIELD)")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Number of environments: {len(GOAL_STATE_ENVS)}")
    print(f"Environments: {', '.join(GOAL_STATE_ENVS)}")
    print(f"Timesteps per environment: {TOTAL_TIMESTEPS:,.0f}")
    print(f"Fixed seed: {FIXED_SEED}")
    print(f"\nShield configuration:")
    print(f"  - Type: DeltaShield")
    print(f"  - Delta: {SHIELD_DELTA}")
    print(f"  - Disable threshold: {REWARD_THRESHOLD} mean reward")
    
    successful = []
    failed = []
    shield_info = {}
    
    for i, env_name in enumerate(GOAL_STATE_ENVS, 1):
        print(f"\n\nProgress: {i}/{len(GOAL_STATE_ENVS)}")
        try:
            disable_timestep = train_environment(env_name)
            successful.append(env_name)
            shield_info[env_name] = disable_timestep
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
        disable_ts = shield_info.get(env_name)
        if disable_ts:
            print(f"  - {env_name} (shield disabled at {disable_ts} timesteps)")
        else:
            print(f"  - {env_name} (shield remained active)")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(GOAL_STATE_ENVS)}")
        for env_name in failed:
            print(f"  - {env_name}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
