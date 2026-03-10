"""
Callback classes for shielding experiments.
This module contains callback implementations for different shield management strategies.
"""

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
    Hard cutoff shielding callback. Removes the shield at a fixed timestep.
    - Disables the shield once num_timesteps >= cutoff_timestep
    """

    def __init__(
        self,
        cutoff_timestep: int = 2_500_000,
        shield_delta: float = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.cutoff_timestep = cutoff_timestep
        self.shield_delta = shield_delta

        self.shield_active = True
        self.actual_cutoff_timestep = None

    def _on_training_start(self) -> None:
        if self.shield_delta is not None:
            self.logger.record("shield/constant_delta", self.shield_delta)

    def _log_shield_stats(self) -> None:
        """Sum blocked/allowed/ignored action counts across all envs and log them."""
        total_blocked = 0
        total_allowed = 0
        total_ignored = 0
        for i in range(self.training_env.num_envs):
            env = self.training_env.envs[i]
            while hasattr(env, 'env') and not isinstance(env, ProbabilisticEnvWrapper):
                env = env.env
            if isinstance(env, ProbabilisticEnvWrapper) and env.shield is not None:
                total_blocked += env.shield.blocked
                total_allowed += env.shield.not_blocked
                total_ignored += getattr(env.shield, 'block_ignored', 0)
        self.logger.record("shield/actions_blocked", total_blocked)
        self.logger.record("shield/actions_allowed", total_allowed)
        self.logger.record("shield/actions_block_ignored", total_ignored)

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        """
        if self.shield_active and self.num_timesteps >= self.cutoff_timestep:
            # Log final shield stats before disabling
            self._log_shield_stats()
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
            self.actual_cutoff_timestep = self.num_timesteps
            self.logger.record("shield_cutoff_timestep", self.num_timesteps)

            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"SHIELD DISABLED at timestep {self.num_timesteps}")
                print(f"   Scheduled cutoff at: {self.cutoff_timestep}")
                print(f"   Continuing training without shield...")
                print(f"{'='*80}\n")

        elif self.shield_active:
            self._log_shield_stats()

        return True


class GradualShieldReductionCallback(BaseCallback):
    """
    Custom callback that gradually reduces shield protection at fixed timestep thresholds
    by either decreasing delta values or increasing ignore_prob.
    """

    def __init__(self,
                 mechanism: str = "delta",
                 initial_delta: float = None,
                 delta_schedule: list[float] = None,
                 ignore_prob_schedule: list[float] = None,
                 ignore_prob_delta: float = None,
                 timestep_schedule: list[float] = None,
                 verbose: int = 1):
        super().__init__(verbose)

        if mechanism not in ["delta", "ignore_prob"]:
            raise ValueError("mechanism must be 'delta' or 'ignore_prob'")

        if timestep_schedule is None:
            timestep_schedule = [1_000_000, 2_000_000, 3_000_000, 4_000_000]

        if mechanism == "delta":
            if delta_schedule is None:
                delta_schedule = [0.8, 0.6, 0.4, 0.2]
            if len(delta_schedule) != len(timestep_schedule):
                raise ValueError("delta_schedule and timestep_schedule must have the same length")
        else:  # ignore_prob
            if ignore_prob_schedule is None:
                ignore_prob_schedule = [0.2, 0.4, 0.6, 0.8]
            if ignore_prob_delta is None:
                raise ValueError(
                    "ignore_prob_delta must be provided when mechanism='ignore_prob'. "
                    "It specifies the fixed delta value of the DeltaShield used throughout training."
                )
            if len(ignore_prob_schedule) != len(timestep_schedule):
                raise ValueError("ignore_prob_schedule and timestep_schedule must have the same length")

        self.mechanism = mechanism
        self.delta_schedule = delta_schedule
        self.ignore_prob_schedule = ignore_prob_schedule
        self.ignore_prob_delta = ignore_prob_delta
        self.timestep_schedule = [int(t) for t in timestep_schedule]
        self.initial_delta = initial_delta if mechanism == "delta" else None

        # Use appropriate schedule based on mechanism
        self.active_schedule = delta_schedule if mechanism == "delta" else ignore_prob_schedule

        # State tracking: current_stage is the index of the next scheduled transition
        self.current_stage = 0
        self.stage_transitions = []  # Track when each transition happened

    def _on_training_start(self) -> None:
        if self.mechanism == "delta" and self.initial_delta is not None:
            self.logger.record("shield/initial_delta", self.initial_delta)
        elif self.mechanism == "ignore_prob" and self.ignore_prob_delta is not None:
            self.logger.record("shield/constant_delta", self.ignore_prob_delta)

        if self.verbose > 0:
            print(f"\nGRADUAL SHIELD REDUCTION SCHEDULE ({self.mechanism.upper()}):")
            if self.mechanism == "ignore_prob":
                print(f"   Fixed δ={self.ignore_prob_delta:.2f} throughout (only ignore_prob varies)")
            for i, (value, ts) in enumerate(zip(self.active_schedule, self.timestep_schedule)):
                if self.mechanism == "delta":
                    print(f"   Stage {i}: δ={value:.2f} at timestep {ts:,}")
                else:
                    print(f"   Stage {i}: ignore_prob={value:.2f} at timestep {ts:,}")
            print()
    
    def _log_shield_stats(self) -> None:
        """Sum blocked/allowed/ignored action counts across all envs and log them."""
        total_blocked = 0
        total_allowed = 0
        total_ignored = 0
        for i in range(self.training_env.num_envs):
            env = self.training_env.envs[i]
            while hasattr(env, 'env') and not isinstance(env, ProbabilisticEnvWrapper):
                env = env.env
            if isinstance(env, ProbabilisticEnvWrapper) and env.shield is not None:
                total_blocked += env.shield.blocked
                total_allowed += env.shield.not_blocked
                total_ignored += getattr(env.shield, 'block_ignored', 0)
        self.logger.record("shield/actions_blocked", total_blocked)
        self.logger.record("shield/actions_allowed", total_allowed)
        self.logger.record("shield/actions_block_ignored", total_ignored)

    def _on_step(self) -> bool:
        """
        Called at every environment step. Transitions to the next shield stage when the
        scheduled timestep is reached.
        """
        while self.current_stage < len(self.timestep_schedule):
            if self.num_timesteps >= self.timestep_schedule[self.current_stage]:
                self._transition_to_stage(self.current_stage)
            else:
                break

        self._log_shield_stats()
        return True  # Continue training
    
    def _transition_to_stage(self, stage: int):
        """Transition to a new shield protection stage."""
        old_value = self.active_schedule[self.current_stage - 1] if self.current_stage > 0 else None
        new_value = self.active_schedule[stage]
        scheduled_ts = self.timestep_schedule[stage]

        # Update the shield in the environment
        self._update_environment_shield(new_value)

        # Track the transition
        transition_data = {
            'timestep': self.num_timesteps,
            'scheduled_timestep': scheduled_ts,
            'stage': stage,
        }
        if self.mechanism == "delta":
            transition_data['delta'] = new_value
        else:
            transition_data['ignore_prob'] = new_value

        self.stage_transitions.append(transition_data)
        self.current_stage = stage + 1  # advance past this stage

        # Log the transition
        self.logger.record("shield/stage", stage)
        self.logger.record(f"shield/{self.mechanism}", new_value)
        self.logger.record("shield/transition_timestep", self.num_timesteps)

        if self.verbose > 0:
            if self.mechanism == "delta":
                prev_str = f"δ={old_value:.2f} → " if old_value is not None else ""
                print(f"\nSHIELD REDUCED at timestep {self.num_timesteps} (scheduled: {scheduled_ts:,})")
                print(f"   Stage {stage}: {prev_str}δ={new_value:.2f}")
                print(f"   Continuing with {'no shield' if new_value == 0.0 else f'delta={new_value:.2f}'}...\n")
            else:  # ignore_prob
                prev_str = f"ignore_prob={old_value:.2f} → " if old_value is not None else ""
                print(f"\nSHIELD IGNORE_PROB INCREASED at timestep {self.num_timesteps} (scheduled: {scheduled_ts:,})")
                print(f"   Stage {stage}: {prev_str}ignore_prob={new_value:.2f}")
                print(f"   Continuing with ignore_prob={new_value:.2f}...\n")
    
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
            print(f"\nSHIELD REDUCTION SUMMARY ({self.mechanism.upper()}):")
            stages_done = len(self.stage_transitions)
            print(f"   Stages completed: {stages_done}/{len(self.active_schedule)}")

            if self.stage_transitions:
                for i, transition in enumerate(self.stage_transitions):
                    param_name = "delta" if self.mechanism == "delta" else "ignore_prob"
                    param_value = transition.get(param_name, "N/A")
                    print(f"     {i+1}. Timestep {transition['timestep']:,}: "
                          f"{param_name}={param_value:.2f} (scheduled: {transition['scheduled_timestep']:,})")
            else:
                print(f"   No stage transitions occurred")