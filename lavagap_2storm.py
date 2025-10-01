from minigrid.envs.lavagap import LavaGapEnv
from minigrid.envs.dynamicobstacles import DynamicObstaclesEnv
from stormvogel import bird, show, ModelType
from minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions

from probabilistic_minigrids import ProbabilisticEnvWrapper

env = LavaGapEnv(5, render_mode="human")
env.reset()
init_dir = env.agent_dir
init_pos =  tuple(map(int, env.agent_pos))
init = (init_pos, init_dir)

def labels(s: bird.State):
    cell_type = "None"
    cell = env.grid.get(s[0][0], s[0][1])
    if cell is not None: 
        cell_type = cell.type
    return f"Position: {s[0]}, Direction: {s[1]}, Cell Type: {cell_type}"

def available_actions(s: bird.State):
    return [["right"], ["left"], ["forward"]]

def delta(s: bird.State, a: bird.Action):
    curr_state = env.grid.get(s[0][0], s[0][1])
    if curr_state is not None and (curr_state.type == "lava" or curr_state.type == "goal"): 
        return [(1, s)]

    if a[0] == "right": 
        result = (s[0], (s[1] + 1) % 4)
        return [(1, result)]
    if a[0] == "left":
        if s[1]-1 < 0: 
            result = (s[0], s[1]-1 + 4)
        else: 
            result = (s[0], s[1]-1)
        return [(1, result)]
    if a[0] == "forward":
        env.agent_pos = s[0]
        env.agent_dir = s[1]
        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(fwd_pos[0], fwd_pos[1])
        result = (tuple(map(int, fwd_pos)), s[1])

        if fwd_cell is None or fwd_cell.can_overlap():
            return [(1,result)]
        if fwd_cell is not None and (fwd_cell.type == "goal" or fwd_cell.type == "lava"):
            return [(1,result)]
        return [(1, s)]



def test_delta(): 
    res1 = delta(init, "forward")
    print(f"delta (action:forward): {res1}")
    res2 = delta(init, "right")
    print(f"delta (action:right): {res2}")
    res3 = delta(init, "left")
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


    manual_control = ManualControl(test_wrapper)
    manual_control.start()


    # visual = show(model, show_editor=True)
    # test_delta()
    # model.states
    # manual_control = ManualControl(env)
    # manual_control.start()



if __name__ == "__main__":
    main()
