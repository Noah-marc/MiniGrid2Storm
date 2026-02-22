"""
Training script for all goal_state environments WITH shield.
The shield protection is gradually reduced by slowly decreasing delta values when performance thresholds are reached.

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
OUTPUT_DIR = project_root / "experiments" / "1.0" / "shielded_gradual_reduction"
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
TOTAL_TIMESTEPS = 5_000_000  # 5e6
FEATURES_DIM = 128
FIXED_SEED = 42
NUM_ENVS = 24  # Number of parallel environments
BATCH_SIZE = 256  # Batch size for PPO

# Shield configuration - gradual reduction schedule
INITIAL_DELTA = 0.9  # Start with high protection
DELTA_SCHEDULE = [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]  # Gradual reduction to no shield
REWARD_THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85]  # Performance thresholds for transitions

# Alternative ignore_prob schedule (probability of ignoring actions)
IGNORE_PROB_SCHEDULE = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]  # Gradual increase in ignoring probability

# Shield mechanism selection
SHIELD_MECHANISM = "delta"  # Options: "delta" or "ignore_prob"


class GradualShieldReductionCallback(BaseCallback):
    """
    Custom callback that monitors training progress and gradually reduces shield protection
    by either decreasing delta values or increasing ignore_prob when performance thresholds are reached.
    """
    
    def __init__(self, 
                 mechanism: str = "delta",
                 delta_schedule: list[float] = DELTA_SCHEDULE,
                 ignore_prob_schedule: list[float] = IGNORE_PROB_SCHEDULE,
                 reward_thresholds: list[float] = REWARD_THRESHOLDS,
                 nr_episodes: int = 100,
                 verbose: int = 1):
        super().__init__(verbose)
        
        # Validation
        if mechanism not in ["delta", "ignore_prob"]:
            raise ValueError("mechanism must be 'delta' or 'ignore_prob'")
        
        if mechanism == "delta" and len(delta_schedule) != len(reward_thresholds):
            raise ValueError("delta_schedule and reward_thresholds must have the same length")
        
        if mechanism == "ignore_prob" and len(ignore_prob_schedule) != len(reward_thresholds):
            raise ValueError("ignore_prob_schedule and reward_thresholds must have the same length")
        
        self.mechanism = mechanism
        self.delta_schedule = delta_schedule
        self.ignore_prob_schedule = ignore_prob_schedule
        self.reward_thresholds = reward_thresholds
        self.nr_episodes = nr_episodes
        
        # Use appropriate schedule based on mechanism
        self.active_schedule = delta_schedule if mechanism == "delta" else ignore_prob_schedule
        
        # State tracking
        self.current_stage = 0  # Which stage in the schedule we're at
        self.ep_rewards = deque(maxlen=self.nr_episodes)
        self.stage_transitions = []  # Track when each transition happened
        
        if self.verbose > 0:
            print(f"\n📊 GRADUAL SHIELD REDUCTION SCHEDULE ({mechanism.upper()}):")
            for i, (value, threshold) in enumerate(zip(self.active_schedule, reward_thresholds)):
                if mechanism == "delta":
                    if i == 0:
                        print(f"   Stage {i}: δ={value:.1f} (initial)")
                    else:
                        print(f"   Stage {i}: δ={value:.1f} (when mean reward ≥ {threshold:.2f})")
                else:  # ignore_prob
                    if i == 0:
                        print(f"   Stage {i}: ignore_prob={value:.1f} (initial)")
                    else:
                        print(f"   Stage {i}: ignore_prob={value:.1f} (when mean reward ≥ {threshold:.2f})")
            print(f"   Tracking performance over {nr_episodes} episodes\n")
    
    def _on_step(self) -> bool:
        """
        Called at every environment step. We check episode completions and update shield parameters.
        """
        # Only process if we haven't reached final stage
        if self.current_stage < len(self.active_schedule) - 1:
            infos = self.locals.get("infos", [])

            for info in infos:
                # Monitor wrapper adds this when an episode ends
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    self.ep_rewards.append(ep_reward)

                    # Only evaluate once we have enough episodes
                    if len(self.ep_rewards) == self.nr_episodes:
                        mean_reward = np.mean(self.ep_rewards)
                        
                        # Check if we should move to next stage
                        next_stage = self.current_stage + 1
                        if next_stage < len(self.reward_thresholds):
                            threshold = self.reward_thresholds[next_stage]
                            
                            if mean_reward >= threshold:
                                self._transition_to_stage(next_stage, mean_reward)
        
        return True  # Continue training
    
    def _transition_to_stage(self, new_stage: int, current_mean_reward: float):
        """Transition to a new shield protection stage."""
        old_value = self.active_schedule[self.current_stage]
        new_value = self.active_schedule[new_stage]
        threshold = self.reward_thresholds[new_stage]
        
        # Update the shield in the environment
        self._update_environment_shield(new_value)
        
        # Track the transition
        transition_data = {
            'timestep': self.num_timesteps,
            'stage': new_stage,
            'mean_reward': current_mean_reward,
            'threshold': threshold
        }
        
        if self.mechanism == "delta":
            transition_data['delta'] = new_value
        else:
            transition_data['ignore_prob'] = new_value
        
        self.stage_transitions.append(transition_data)
        self.current_stage = new_stage
        
        # Log the transition
        self.logger.record(f"shield/stage", new_stage)
        self.logger.record(f"shield/{self.mechanism}", new_value)
        self.logger.record(f"shield/transition_timestep", self.num_timesteps)
        
        if self.verbose > 0:
            if self.mechanism == "delta":
                if new_value == 0.0:
                    print(f"\n🎯 SHIELD COMPLETELY DISABLED at timestep {self.num_timesteps}")
                    print(f"   Stage {new_stage}: δ={old_value:.1f} → δ={new_value:.1f} (NO SHIELD)")
                else:
                    print(f"\n⬇️  SHIELD PROTECTION REDUCED at timestep {self.num_timesteps}")
                    print(f"   Stage {new_stage}: δ={old_value:.1f} → δ={new_value:.1f}")
                    
                print(f"   Mean reward achieved: {current_mean_reward:.3f} ≥ {threshold:.2f}")
                print(f"   Continuing with {'no shield' if new_value == 0.0 else f'δ={new_value:.1f}'}...\n")
            
            else:  # ignore_prob
                if new_value == 1.0:
                    print(f"\n🎯 SHIELD EFFECTIVELY DISABLED at timestep {self.num_timesteps}")
                    print(f"   Stage {new_stage}: ignore_prob={old_value:.1f} → {new_value:.1f} (IGNORING ALL)")
                else:
                    print(f"\n⬆️  SHIELD IGNORE PROBABILITY INCREASED at timestep {self.num_timesteps}")
                    print(f"   Stage {new_stage}: ignore_prob={old_value:.1f} → {new_value:.1f}")
                
                print(f"   Mean reward achieved: {current_mean_reward:.3f} ≥ {threshold:.2f}")
                print(f"   Continuing with {'effectively no shield' if new_value == 1.0 else f'ignore_prob={new_value:.1f}'}...\n")
    
    def _update_environment_shield(self, new_value: float):
        """Update the shield parameter in all vectorized environments."""
        try:
            # Access the vectorized environment
            vec_env = self.training_env
            
            # Update shield parameter for all environments in the vector
            for i in range(vec_env.num_envs):
                env = vec_env.envs[i]
                
                # Navigate through wrapper layers to find ProbabilisticEnvWrapper
                while hasattr(env, 'env') and not isinstance(env, ProbabilisticEnvWrapper):
                    env = env.env
                
                if isinstance(env, ProbabilisticEnvWrapper):
                    if self.mechanism == "delta":
                        if new_value == 0.0:
                            env.remove_shield()
                        else:
                            # Update the delta in the shield
                            if hasattr(env, 'shield') and env.shield is not None:
                                env.shield.delta = new_value
                            else:
                                # Recreate shield with new delta if needed
                                env.reset()
                                model, _ = env.env.unwrapped.convert_to_probabilistic_storm()
                                new_shield = DeltaShield(model, "Pmin=? [F \"lava\"]", delta=new_value)
                                env.set_shield(new_shield)
                    
                    else:  # ignore_prob
                        # Update the ignore_prob in the shield
                        if hasattr(env, 'shield') and env.shield is not None:
                            env.shield.ignore_prob = new_value
                        else:
                            # Should not happen unless shield was removed
                            print(f"Warning: No shield found in env {i} when trying to set ignore_prob")
                else:
                    raise RuntimeError(f"Could not find ProbabilisticEnvWrapper in env {i}")
                    
        except Exception as e:
            print(f"Error updating environment shield parameter: {e}")
            if self.verbose > 0:
                import traceback
                traceback.print_exc()
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print(f"\n📈 SHIELD REDUCTION SUMMARY ({self.mechanism.upper()}):")
            print(f"   Final stage: {self.current_stage}/{len(self.active_schedule)-1}")
            
            final_value = self.active_schedule[self.current_stage]
            if self.mechanism == "delta":
                print(f"   Final delta: {final_value:.1f}")
            else:
                print(f"   Final ignore_prob: {final_value:.1f}")
            
            if self.stage_transitions:
                print(f"   Transitions made: {len(self.stage_transitions)}")
                for i, transition in enumerate(self.stage_transitions):
                    param_name = "delta" if self.mechanism == "delta" else "ignore_prob"
                    param_value = transition.get(param_name, "N/A")
                    print(f"     {i+1}. Timestep {transition['timestep']}: "
                          f"{param_name}={param_value:.1f} (reward: {transition['mean_reward']:.3f})")
            else:
                print(f"   No stage transitions occurred")


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


def plot_training_results(log_dir: Path, env_name: str, output_path: Path, shield_disable_timestep=None):
    """Load training results and create performance plots."""
    try:
        # Read progress.csv from PPO logger
        progress_file = log_dir / "progress.csv"
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


def save_env_image(env, env_name: str, output_path: Path):
    """Render and save an image of the environment."""
    try:
        # Reset to get initial state
        env.reset()
        # Access the unwrapped environment to render properly
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
        
        # Add shield with initial delta
        shield = DeltaShield(INITIAL_DELTA)
        env = ProbabilisticEnvWrapper(env, shield)
        
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    
    # Add monitoring to track episode statistics for logging
    env = VecMonitor(env, filename=str(log_dir / "monitor"))
    
    print(f"   {NUM_ENVS} environments created with shield (δ={INITIAL_DELTA})")
    
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
    
    # Create callback for gradual shield reduction
    shield_callback = GradualShieldReductionCallback(
        mechanism=SHIELD_MECHANISM,
        delta_schedule=DELTA_SCHEDULE,
        ignore_prob_schedule=IGNORE_PROB_SCHEDULE,
        reward_thresholds=REWARD_THRESHOLDS,
        nr_episodes=100,
        verbose=1
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


