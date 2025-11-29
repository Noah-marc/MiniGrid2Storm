from probabilistic_minigrids import ProbabilisticEnvWrapper
from stormvogel.stormpy_utils.model_checking import model_checking
from minigrid.core.actions import Actions
from stormvogel.model import Action
from PIL import Image

from utils import load_env_configs, process_config


configs = load_env_configs("./goal_state_envs.yaml")
env_info = process_config(configs[0])
# Add render_mode parameter to the environment parameters
env_instance = env_info['env_class'](**env_info['env_params'])
env = ProbabilisticEnvWrapper(env_instance, env_info['used_actions'], env_info['prob_distribution'])


model, visited_envs = env.convert_to_probabilistic_storm()
safety_property_string = f"Pmin=? [F \"lava\"]"
min_probs = model_checking(model, safety_property_string)
delta = 0.5


def delta_shield(state, action): 
    assert state is not None
    assert action is not None

    act_val = action_value(state, action)
    optimal_act_val = min_probs.values[state]
    if delta * act_val <= optimal_act_val:
        return action
    else:
        print(f"Action {action.name} blocked by delta-shielding in state {state}.")



def action_value(state:int, action:Actions): 
    """ Compute the value of taking action in state."""

    choice_for_state = model.choices[state]
    
    # Create Action object directly - much more efficient!
    stormvogel_action = Action(frozenset({action.name}))
    
    # Direct dictionary lookup - O(1) instead of O(n) iteration
    if stormvogel_action not in choice_for_state.transition:
        raise ValueError(f"Action {action.name} not found in state {state}")
    
    subranches = choice_for_state.transition[stormvogel_action]
    value = 0.0

    for prob_s_prime, s_prime in subranches.branch: 
        min_prob_s_prime = min_probs.values[s_prime.id]
        value += prob_s_prime * min_prob_s_prime

    return value

def main(): 
    # Test the efficient action_value function
    print("Testing efficient action_value:")
    print(f"Right action value: {action_value(0, Actions.right)}")
    print(f"Forward action value: {action_value(0, Actions.forward)}")
    print(f"Left action value: {action_value(0, Actions.left)}")
    img = Image.fromarray(env.render())
    img.show()

if __name__ == "__main__":
    main()








    

    #2. Compute value of all actions from state, state and get optimal value
    #3. Check if chosen action's value fulfills delta*val(state,action) <= optimal_value.
    #4. If 3 holds, then allow action by returning the action. If not, return ... 



    pass

    
    
