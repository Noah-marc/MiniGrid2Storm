from stormvogel.stormpy_utils.model_checking import model_checking
from prob_minigrid2storm import convert_to_probabilistic_storm, load_env_configs, process_config
from stormvogel.result import Result
from stormvogel.extensions.visual_algos import policy_iteration
from probabilistic_minigrids import ProbabilisticEnvWrapper


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
    
    # You could also try different objectives:
    # safety_maximization = "Pmax=? [G !\"lava\"]"  # Maximize probability of never hitting lava
    # lava_minimization = "Pmin=? [F \"lava\"]"     # Minimize probability of hitting lava


if __name__ == "__main__":
    # test_model_checking(crossing_env_storm, "Probabilistic CrossingEnv")
    # test_policy_iteration()

    distshift_env_prob = ProbabilisticEnvWrapper(distshift_env_instance, distshift_env_info['used_actions'], distshift_env_info['prob_distribution'])
    model, visited_states = distshift_env_prob.convert_to_probabilistic_storm()
    print(f"Converted DistShiftEnv to probabilistic storm model with {len(model.states)} states.")
    print(f"Visited states: {list(visited_states.keys())}")