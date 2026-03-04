"""
Training utilities for MiniGrid experiments.
This module contains common utility functions shared across training scripts.
"""

from pathlib import Path
from PIL import Image


def make_video_trigger(recording_timesteps: list, num_envs: int):
    """
    Returns a trigger function for VecVideoRecorder based on total timesteps.
    VecVideoRecorder passes step_id (vectorized step count) to the trigger;
    total_timesteps = step_id * num_envs.
    """
    triggered = set()
    tolerance = num_envs * 4  # Window wide enough to catch the target between steps

    def trigger(step_id: int) -> bool:
        total_ts = step_id * num_envs
        for ts in recording_timesteps:
            if ts not in triggered and ts <= total_ts <= ts + tolerance:
                triggered.add(ts)
                return True
        return False

    return trigger


def save_env_image(env, env_name: str, output_path: Path):
    """Render and save an image of the environment."""
    try:
        # Reset to get initial state
        env.reset()
        # Access the unwrapped environment to render properly
        # Try different unwrapping approaches to handle various wrapper configurations
        try:
            # First try direct access through wrapper
            base_env = env.env
        except AttributeError:
            try:
                # Try unwrapped for more complex wrapper chains
                base_env = env.unwrapped.env
            except AttributeError:
                # Fall back to the env itself
                base_env = env
        
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