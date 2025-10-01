"""
This code converts a probabilistic lavagap environment to a storm model. 
We used the probabilistic wrapper from probabilistic_minigrids.py to create a probabilistic lavagap environment. 
"""

from minigrid.envs.lavagap import LavaGapEnv
from stormvogel import bird, show, ModelType
from minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions

from probabilistic_minigrids import ProbabilisticEnvWrapper


env = LavaGapEnv(5, render_mode="human")
used_actions = [Actions.forward, Actions.left, Actions.right]
#we store the indices of the actions to be able to use them later. Format is {action: index}
action_indices = {action: i for i, action in enumerate(used_actions)}
prob_distribution = {Actions.forward : [0.8, 0.1, 0.1], 
                     Actions.left : [0.1, 0.8, 0.1], 
                     Actions.right : [0.1, 0.1, 0.8]}
prob_env = ProbabilisticEnvWrapper(env, used_actions=used_actions, prob_distribution=prob_distribution)
prob_env.reset()
init_dir = prob_env.agent_dir
init_pos =  tuple(map(int, prob_env.agent_pos))
init = (init_pos, init_dir)


def labels(s: bird.State):
    cell_type = "None"
    cell = prob_env.grid.get(s[0][0], s[0][1])
    if cell is not None: 
        cell_type = cell.type
    return f"Position: {s[0]}, Direction: {s[1]}, Cell Type: {cell_type}"

def available_actions(s: bird.State):
    """
    Up until now it is assumed that all actions are always available. 
    TODO: Change this if needed.
    """
    return [[action.name] for action in used_actions] # We need to reformat for bird api

def delta(s: bird.State, a: bird.Action):
    """Up until know it is assumed that the actions are only forward, left, right. Change this later. if needed"""  
    # Use consistent environment - prob_env throughout
    curr_state = prob_env.grid.get(s[0][0], s[0][1])
    if curr_state is not None and (curr_state.type == "lava" or curr_state.type == "goal"): 
        return [(1, s)]

    result = []
    action = getattr(Actions, a[0])
    probs = prob_env.prob_distribution[action]

    #First we store the results for each action
    result_right = (s[0], (s[1] + 1) % 4)
    result.append((probs[action_indices[Actions.right]], result_right))
    if s[1]-1 < 0: 
        result_left = (s[0], s[1]-1 + 4)
    else: 
        result_left = (s[0], s[1]-1)
    result.append((probs[action_indices[Actions.left]], result_left))

    # Handle forward movement with proper collision detection
    prob_env.agent_pos = s[0]
    prob_env.agent_dir = s[1]
    fwd_pos = prob_env.front_pos
    fwd_cell = prob_env.grid.get(fwd_pos[0], fwd_pos[1])
    
    # Apply the same logic as the original version
    if fwd_cell is None or fwd_cell.can_overlap():
        result_forward = (tuple(map(int, fwd_pos)), s[1])
        result.append((probs[action_indices[Actions.forward]], result_forward))
    elif fwd_cell is not None and (fwd_cell.type == "goal" or fwd_cell.type == "lava"):
        result_forward = (tuple(map(int, fwd_pos)), s[1])
        result.append((probs[action_indices[Actions.forward]], result_forward))
    else:
        # Can't move forward (wall/obstacle), stay in same position
        result.append((probs[action_indices[Actions.forward]], s))
    return result

def test_delta(): 
    res1 = delta(init, ["forward"])
    print(f"delta (action:forward): {res1}")
    res2 = delta(init, ["right"])
    print(f"delta (action:right): {res2}")
    res3 = delta(init, ["left"])
    print(f"delta (action:left): {res3}")


def main():
    pass
    # model = bird.build_bird(
    #                 delta=delta,
    #                 init=init, 
    #                 labels=labels, 
    #                 available_actions=available_actions, 
    #                 modeltype=ModelType.MDP
    #                 )
    # print(len(model.states))
    # visual = show(model, show_editor=True)

    # test_delta()
    # model.states
    manual_control = ManualControl(prob_env)
    manual_control.start()



if __name__ == "__main__":
    main()
