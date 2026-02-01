from minigrid.core.actions import Actions
from typing import List, Dict
from time import sleep
from stormvogel.extensions.visual_algos import arg_max 
from stormvogel.model import Action as StormvogelAction

import yaml
import importlib
import stormvogel




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


def custom_policy_iteration(model: stormvogel.model.Model,
    prop: str,
    visualize: bool = True,
    layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
    delay: int = 2,
    clear: bool = True,): 
    """ Basically the same as policy_iteration from visual_algos.py, but returns the final scheduler as well. """
    
    old = None
    new = stormvogel.random_scheduler(model)

    while not old == new:
        old = new

        dtmc = old.generate_induced_dtmc()
        dtmc_result = stormvogel.model_checking(dtmc, prop=prop)  # type: ignore

        if visualize:
            vis = stormvogel.visualization.JSVisualization(
                model, layout=layout, scheduler=old, result=dtmc_result
            )
            vis.show()
            sleep(delay)
            if clear:
                vis.clear()

        choices = {
            i: arg_max(
                [
                    lambda a: sum(
                        [
                            (p * dtmc_result.get_result_of_state(s2.id))  # type: ignore
                            for p, s2 in s1.get_outgoing_choice(a)  # type: ignore
                        ]
                    )
                    for _ in s1.available_actions()
                ],
                s1.available_actions(),
            )
            for i, s1 in model
        }
        new = stormvogel.Scheduler(model, choices)
    if visualize:
        print("Value iteration done:")
        stormvogel.show(model, layout=layout, scheduler=new, result=dtmc_result)  # type: ignore
    return dtmc_result, new  # type: ignore

# Note: Action conversion functions have been moved to action_utils.py
# Use from_stormvogel_action, to_stormvogel_action, etc. from that module
