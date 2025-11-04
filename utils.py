from minigrid.core.actions import Actions
from typing import List, Dict
import yaml
import importlib




def load_env_configs(config_path = "./environments.yaml") -> List[Dict]: 
    """Loads environment configurations from environments.yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config['environments']


def process_config(env_config: Dict) -> Dict:
    """
    Processes parameters for environment creation from a single environment config
    """

    env_module = importlib.import_module(env_config['module'])
    env_class = getattr(env_module, env_config['class'])
    used_actions = [getattr(Actions, action) for action in env_config['actions']]
    prob_distribution = {
                getattr(Actions, action): probs for action, probs in env_config['probabilities'].items()
            }

    return {'env_name': env_config['name'], 'env_class': env_class,'env_params': env_config.get('class_params', {}), 'used_actions': used_actions, 'prob_distribution': prob_distribution}
