"""
In this file, we define the actual simulations using a shield and an probabilistic 
minigrid environment.
"""
from stormvogel import Scheduler, Model
from shield import Shield
from probabilistic_minigrids import ProbabilisticEnvWrapper
from action_utils import from_stormvogel_action


def run_simulation(env: ProbabilisticEnvWrapper, policy:Scheduler, num_episodes: int = 10, shield:Shield = None):
    """ 
    This function runs a simulation of the given environment using the provided policy for a specified number of episodes.
    If a shield is provided, it will be used to filter actions before executing them in the environment.
    

    Args: 
        env (ProbabilisticEnvWrapper): The probabilistic MiniGrid environment to simulate.
        policy (Scheduler): The policy to use for action selection.
        num_episodes (int): The number of episodes to simulate. Default is 10.
        shield (Shield, optional): An optional shield to filter actions. Default is None.
    """
    
    if not hasattr(env, 'hash_to_state_id'):
        if not Shield: 
            raise Warning("Environment has not been converted to StormVogel yet, but you provided a shield. This might lead to inconsistencies. ")
        env.convert_to_probabilistic_storm()

    # Use a fixed seed to ensure consistent environment layout across episodes
    SIMULATION_SEED = env.env.np_random_seed
    reach_goal = 0
    reach_lava = 0
    reach_truncated = 0
    
    for episode in range(num_episodes): 
        # Note that the reset for the 1. iteration is unnecearry, as the environment is already reset in the probabilisticEnvWrapper and we just use the seed generated during that reset.
        env.reset(seed=SIMULATION_SEED)  # Same seed = same layout = same hash_to_state_id mapping
        terminated = False
        truncated = False
        current_state = env.get_current_state_id()  
        step_count = 0
        
        while not (terminated or truncated):
            action = policy.get_choice_of_state(current_state)
            if shield is not None:
                action = shield.verify_action(current_state, action)

            action = from_stormvogel_action(action)
            current_state = env.get_current_state_id()
        
        agent_pos = env.env.agent_pos
        agent_cell = env.env.grid.get(agent_pos[0], agent_pos[1])
        agent_cell_type = agent_cell.type if agent_cell is not None else "None"
        
        if terminated:
            if reward > 0:
                # Validate that agent is actually on a goal cell
                if agent_cell_type != "goal":
                    raise ValueError(f"Episode {episode}: Terminated with positive reward but agent is on '{agent_cell_type}' cell, not 'goal' at position {agent_pos}")
                reach_goal += 1
            else:
                # Validate that agent is actually on a lava cell
                if agent_cell_type != "lava":
                    raise ValueError(f"Episode {episode}: Terminated with non-positive reward but agent is on '{agent_cell_type}' cell, not 'lava' at position {agent_pos}")
                reach_lava += 1
        elif truncated:
            # Validate that truncation is due to max steps
            max_steps = getattr(env.env, 'max_steps', None)
            if max_steps is not None and step_count < max_steps:
                raise ValueError(f"Episode {episode}: Truncated after {step_count} steps but max_steps is {max_steps}")
            reach_truncated += 1
    
    if shield:
        print(f"DEBUG: Shield blocked {shield.blocked} actions and validated {shield.not_blocked}")
        shield.blocked = 0
        shield.not_blocked = 0 
    return reach_goal, reach_lava, reach_truncated