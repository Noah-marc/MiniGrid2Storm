"""
This code converts a probabilistic lavagap environment to a storm model. 
We used the probabilistic wrapper from probabilistic_minigrids.py to create a probabilistic lavagap environment. 
"""

from minigrid.envs.lavagap import LavaGapEnv
from stormvogel import bird, show, ModelType
from minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
import copy

from probabilistic_minigrids import ProbabilisticEnvWrapper

env = LavaGapEnv(5, render_mode=None)
used_actions = [Actions.forward, Actions.left, Actions.right]
prob_distribution = {Actions.forward : [0.8, 0.1, 0.1], 
                     Actions.left : [0.1, 0.8, 0.1], 
                     Actions.right : [0.1, 0.1, 0.8]}
prob_env = ProbabilisticEnvWrapper(env, used_actions=used_actions, prob_distribution=prob_distribution)
prob_env.reset()
init_dir = prob_env.agent_dir
init_pos =  tuple(map(int, prob_env.agent_pos))

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
    return f"Position: {(int(curr_env.agent_pos[0]),int(curr_env.agent_pos[1]))}, Direction: {curr_env.agent_dir}, Cell Type: {cell_type}"

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
    for action,i in enumerate(used_actions): 
        env_copy = copy.deepcopy(curr_env)
        env_copy.step(action)
        hash = env_copy.hash()
        visited_envs[hash] = env_copy
        result.append((probs[i], hash))
    return result

def main():
    model = bird.build_bird(
                    delta=delta,
                    init=init, 
                    labels=labels, 
                    available_actions=available_actions, 
                    modeltype=ModelType.MDP
                    )
    
    # print(len(model.states))
    # visual = show(model, show_editor=True)

    # test_delta()
    # model.states
    # manual_control = ManualControl(prob_env)
    # manual_control.start()



if __name__ == "__main__":
    main()
