""" 
Responsible for loading and wrapping environments based on configuration files. 
"""

import yaml
import importlib
from probabilistic_minigrids import ProbabilisticEnvWrapper
from minigrid.core.actions import Actions

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_wrapped_env_from_config(config: dict)-> ProbabilisticEnvWrapper:

    env_module = importlib.import_module(config['module'])
    env_class = getattr(env_module, config['class'])
    used_actions = [getattr(Actions, action) for action in config['actions']]
    prob_distribution = {
                getattr(Actions, action): probs for action, probs in config['probabilities'].items()
            }

    base_env = env_class(**config['class_params'])
    wrapped_env = ProbabilisticEnvWrapper(base_env, used_actions, prob_distribution)

    return wrapped_env