
from stormvogel.stormpy_utils.model_checking import model_checking
from stormvogel.result import Result
from stormvogel.extensions.visual_algos import policy_iteration
from pathlib import Path
import logging
from PIL import Image

from probabilistic_minigrids import ProbabilisticEnvWrapper
from utils import load_env_configs, process_config


logging.basicConfig(

    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('minigrid2storm.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

configs = load_env_configs()
#contains all the information about the environments that is needed for making a probabilistic env wrapper and converting to storm model.
crossing_env_info= process_config(configs[1])
crossing_env_instance= crossing_env_info['env_class'](**crossing_env_info['env_params'])
crossing_env_prob = ProbabilisticEnvWrapper(crossing_env_instance, crossing_env_info['used_actions'], crossing_env_info['prob_distribution'])
crossing_env_storm, crossing_env_visited_states = crossing_env_prob.convert_to_probabilistic_storm()

distshift_env_info= process_config(configs[2])
distshift_env_instance= distshift_env_info['env_class'](**distshift_env_info['env_params'])
distshift_env_prob = ProbabilisticEnvWrapper(distshift_env_instance, distshift_env_info['used_actions'], distshift_env_info['prob_distribution'])
distshift_env_storm, distshift_visited_states = distshift_env_prob.convert_to_probabilistic_storm()


def test_model_checking(env_storm, env_name): 
    print("=== Model Info ===")
    print(f"Number of states: {len(env_storm.states)}")
    print(f"Available labels: {env_storm.get_labels()}")

    # Example safety properties
    lava_eventually_min = "Pmin=? [F \"lava\"]"    # Probability of eventually reaching lava
    # lava_eventually_max = "Pmax=? [F \"lava\"]"    # Probability of eventually reaching lava
    lava_never_min = "Pmin=? [G !\"lava\"]"        # Probability of never reaching lava
    # lava_never_max = "Pmax=? [G !\"lava\"]"        # Probability of never reaching lava
    goal_eventually_min= "Pmin=? [F \"goal\"]"    # Probability of eventually reaching goal (worst case)
    # goal_eventually_max= "Pmax=? [F \"goal\"]"    # Probability of eventually reaching goal (best case) 

    """ The code below can only be run in a jupyter notebook environment"""
    print("\n=== Safety Properties ===")
    print(f"Testing property: {lava_eventually_min}")
    result:Result = model_checking(env_storm, lava_eventually_min)
    print(result.values)
    # print(result.model.states)
    print(f"times of 1 in results: {[val for _, val in result.values.items() if val == 1]}")
    # print(f"\nTesting property: {lava_eventually_max}")
    # result:Result = model_checking(env_storm, lava_eventually_max) 
    # print(result.maximum_result())

    print(f"\nTesting property: {lava_never_min}")
    result:Result = model_checking(env_storm, lava_never_min)
    print(result.values)
    print(f"times of 1 in results: {[val for _, val in result.values.items() if val == 1]}")
    # print(f"\nTesting property: {lava_never_max}")
    # result:Result = model_checking(env_storm, lava_never_max)
    # print(result.maximum_result())

    print(f"\nTesting property: {goal_eventually_min}")
    result:Result = model_checking(env_storm, goal_eventually_min)
    print(result.values)
    print(f"times of 1 in results: {[val for _, val in result.values.items() if val == 1]}")
    # print(f"\nTesting property: {goal_eventually_max}")
    # result:Result = model_checking(env_storm, goal_eventually_max)
    # print(result.values)


def test_policy_iteration(): 
    # Define what we want to optimize for
    goal_maximization = "P=? [F \"goal\"]"  # Maximize probability of reaching goal
    
    print(f"Running policy iteration to optimize: {goal_maximization}")
    print(f"This will find the policy that maximizes the chance of reaching the goal")
    
    result = policy_iteration(distshift_env_storm, prop=goal_maximization, visualize=False)
    print(f"Optimization result: {result.values}")
    

def load_and_convert_all_envs(): 
    """This function loads all environments from environments.yaml, converts them to probabilistic storm models, and logs the results.

    This is a good function for testing whether loading all envs works. Of course, on should inspect the ouput models further to ensure correctness.

    This function can usually be in utils and then used in your code, but for now I make it a test function.

    """
    env_configs = load_env_configs()
    failed_envs = []
    successful_envs = []
    
    for env_config in env_configs: 
        env_name = env_config['name']
        try:
            logger.debug("Processing new config ...")
            processed_config = process_config(env_config)
            logger.debug(f"Processed config for environment {processed_config['env_name']}")
            env_instance = processed_config['env_class'](**processed_config['env_params'])
            logger.debug("Converting to probabilistic storm model ...")
            prob_env = ProbabilisticEnvWrapper(env_instance, processed_config['used_actions'], processed_config['prob_distribution'])
            model, envs = prob_env.convert_to_probabilistic_storm()
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


def test_add_lava_all_envs(dir_path: str = "./logs/img/"): 
    logger.info("\n\n ==== TESTING  add_lava() FOR ALL ENVS ===")
    env_configs = load_env_configs("./goal_state_envs.yaml")
    failed_envs = []
    successful_envs = []
    for env_config in env_configs: 
        env_name = env_config['name']
        try:
            processed_config = process_config(env_config)
            processed_config['env_params']['render_mode'] = 'rgb_array'

            env_instance = processed_config['env_class'](**processed_config['env_params'])
            prob_env = ProbabilisticEnvWrapper(env_instance, processed_config['used_actions'], processed_config['prob_distribution'])
            pos = prob_env.add_lava()
            img = Image.fromarray(prob_env.render())
            img_path = Path(dir_path).joinpath(f'{env_name}.png')
            img.save(img_path)
            logger.info(f"Image saved to: {img_path}")
            if pos is not None:
                logger.info(f"Lava added to environment {env_name} at position {pos}.")
                successful_envs.append(env_name)
            else:
                logger.info(f"No lava added to environment {env_name} (already has lava or no valid position).")
                failed_envs.append(env_name)
        except Exception as e:
            logger.error(f"ERROR while adding lava to environment {env_name}: {str(e)}")
            continue
    logger.info(f"\n=== SUMMARY of add_lava tests ===")
    logger.info(f"Successful lava additions: {len(successful_envs)}")
    logger.info(f"Failed lava additions: {len(failed_envs)}")

    

def main():
    # test_model_checking(distshift_env_storm, "Probabilistic DistShiftEnv")
    # test_policy_iteration()
    # load_and_convert_all_envs() 
    # load_and_convert_all_envs()
    test_add_lava_all_envs()


if __name__ == "__main__":
    main()