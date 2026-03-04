"""
Callback classes for shielding experiments.
This module contains callback implementations for different shield management strategies.
"""

import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

# Add parent directory to path to import project modules
import sys
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from shield import DeltaShield
from probabilistic_minigrids import ProbabilisticEnvWrapper


class ShieldHardCutoffCallback(BaseCallback):
    """
    Hard cutoff shielding callback. Removes the shield once for 20 episodes the mean reward is above the threshold
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
                            # Access all environments in the vectorized wrapper and disable shields
                            vec_env = self.training_env
                            for i in range(vec_env.num_envs):
                                env = vec_env.envs[i]
                                # Unwrap through wrappers until we reach ProbabilisticEnvWrapper
                                while hasattr(env, 'env') and not isinstance(env, ProbabilisticEnvWrapper):
                                    env = env.env
                                # Now we have the ProbabilisticEnvWrapper
                                if isinstance(env, ProbabilisticEnvWrapper):
                                    env.remove_shield()
                                else:
                                    raise RuntimeError(f"Could not find ProbabilisticEnvWrapper in env {i} to remove shield")
                            
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


class GradualShieldReductionCallback(BaseCallback):
    """
    Custom callback that monitors training progress and gradually reduces shield protection
    by either decreasing delta values or increasing ignore_prob when performance thresholds are reached.
    """
    
    def __init__(self, 
                 mechanism: str = "delta",
                 delta_schedule: list[float] = None,
                 ignore_prob_schedule: list[float] = None,
                 reward_thresholds: list[float] = None,
                 nr_episodes: int = 100,
                 verbose: int = 1):
        super().__init__(verbose)
        
        # Default schedules if not provided
        if delta_schedule is None:
            delta_schedule = [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
        if ignore_prob_schedule is None:
            ignore_prob_schedule = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        if reward_thresholds is None:
            reward_thresholds = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85]
        
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
                                model, _ = env.convert_to_probabilistic_storm()
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