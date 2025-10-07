"""
This code converts a probabilistic lavagap environment to a storm model. 
We used the probabilistic wrapper from probabilistic_minigrids.py to create a probabilistic lavagap environment. 
"""

from stormvogel import bird, show, ModelType
from minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from typing import List, Dict
import copy
import yaml
import importlib
import logging

from probabilistic_minigrids import ProbabilisticEnvWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('minigrid2storm.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

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


def convert_to_probabilistic_storm(env:MiniGridEnv, used_actions:list[Actions], prob_distribution: dict[Actions,list[float]]):


    prob_env = ProbabilisticEnvWrapper(env, used_actions=used_actions, prob_distribution=prob_distribution)
    prob_env.reset()

    #We store states as hashes of envs using the MiniGridEnv.hash function, which returns the same hashes for deepcopies of same envs. When storing states as envs, the bird api will use the __hash__ and __eq__ functions of the env, which do not have the desired properties for deepcopies.
    init = prob_env.env.hash() 
    #We use the dict to get an env from a hash in the delta function. This way, we can simulate steps by using the defined step() functions. 
    visited_envs = {init: prob_env.env}

    def labels(s: str):
        """
        In init we define the inital state of the agent as the env of the probablisitc wrapper. 
        So while theoritcially the type of s would be bird.State, it is actually a MiniGridEnv.
        """
        curr_env = visited_envs[s]
        cell_type = "None"
        cell = curr_env.grid.get(curr_env.agent_pos[0], curr_env.agent_pos[1])
        if cell is not None: 
            cell_type = cell.type
        label = f"Position: {(int(curr_env.agent_pos[0]),int(curr_env.agent_pos[1]))}, Direction: {curr_env.agent_dir}, Cell Type: {cell_type}"
        if Actions.pickup in used_actions:
            carrying = "None"
            if curr_env.carrying is not None:
                carrying = curr_env.carrying.type
            label += f", Carrying: {carrying}"
        return label

    def available_actions(s: str):
        """
        Up until now it is assumed that all actions are always available. 
        TODO: Change this if needed.
        """
        # We reformat the used_action list to the format expected by the bird api
        return [[action.name] for action in used_actions]

    def delta(s: str, a: bird.Action):
        """
        In init we define the inital state of the agent as the env of the probablisitc wrapper. 
        So while theoritcially the type of s would be bird.State, it is actually a MiniGridEnv.
        """
        curr_env = visited_envs[s]
        curr_state = curr_env.grid.get(curr_env.agent_pos[0], curr_env.agent_pos[1])
        if curr_state is not None and (curr_state.type == "lava" or curr_state.type == "goal"): 
            return [(1, s)]
        
        result = []
        given_action = getattr(Actions, a[0])
        probs = prob_env.prob_distribution[given_action]
        for i,action in enumerate(used_actions): 
            env_copy = copy.deepcopy(curr_env)
            env_copy.step(action)
            hash = env_copy.hash()
            visited_envs[hash] = env_copy
            result.append((probs[i], hash))
        return result
    
    model = bird.build_bird(
                    delta=delta,
                    init=init, 
                    labels=labels, 
                    available_actions=available_actions, 
                    modeltype=ModelType.MDP
                    )
    return model
    

def main():
    env_configs = load_env_configs()
    failed_envs = []
    successful_envs = []
    
    for env_config in env_configs: 
        env_name = env_config['name']
        try:
            logger.info("Processing new config ...")
            processed_config = process_config(env_config)
            logger.info(f"Processed config for environment {processed_config['env_name']}")
            env_instance = processed_config['env_class'](**processed_config['env_params'])
            logger.info("Converting to probabilistic storm model ...")
            model = convert_to_probabilistic_storm(env_instance, processed_config['used_actions'], processed_config['prob_distribution'])
            logger.info(f"Model for environment {processed_config['env_name']} loaded to storm successfully.")
            successful_envs.append(env_name)
        except Exception as e:
            logger.error(f"ERROR with environment {env_name}: {str(e)}")
            failed_envs.append((env_name, str(e)))
            continue
    
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Successful environments ({len(successful_envs)}): {successful_envs}")
    logger.info(f"Failed environments ({len(failed_envs)}):")
    for env_name, error in failed_envs:
        logger.info(f"  - {env_name}: {error}")



    # convert_to_probabilistic_storm()
    
    
    # print(len(model.states))
    # visual = show(model, show_editor=True)

    # test_delta()
    # model.states
    # manual_control = ManualControl(prob_env)
    # manual_control.start()



if __name__ == "__main__":
    main()
