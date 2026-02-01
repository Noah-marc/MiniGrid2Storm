"""
Responsible for registering an environtment to gym, based on a config file. 
"""

import gymnasium as gym
from envs.loader import create_wrapped_env_from_config, load_config


def register_env(config_path: str, version: int = 0):
    """
    Register a single environment config (wrapped with the ProbabilisticEnvWrapper) with Gym. 

    config_path: path to YAML file
    version: gym environment version (e.g. 0 → '-v0')
    """
    cfg = load_config(config_path)

    env_id = f"{cfg['name']}-v{version}"

    gym.register(
        id=env_id,
        entry_point="envs.registry:make_env",
        kwargs={"config": cfg},
    )

    print(f"[Gym] Registered environment: {env_id}")

    return env_id


def make_env(config: dict):
    """Entry point called by gym.make()."""
    return create_wrapped_env_from_config(config)

if __name__ == "__main__":
    # Test for crossingEnv
    register_env("./envs/configs/goal_state/CrossingEnv.yaml")