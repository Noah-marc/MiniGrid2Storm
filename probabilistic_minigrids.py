from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ActType
from minigrid.core.actions import Actions
from minigrid.core.world_object import Lava
from stormvogel import bird, ModelType
import copy
import numpy as np
import gymnasium as gym



class ProbabilisticEnvWrapper(gym.Env):
    """
    A wrapper for MiniGrid environments that introduces probabilistic action outcomes.

    Args: 
        env (MiniGridEnv): The MiniGrid environment to wrap.
        used_actions (list[Actions], optional): List of actions that are used from the minigrid env. Defaults to all actions.
        prob_distribution (dict[list[float]], optional): A dictionary mapping each action in used_actions to a list of probabilities for each action in used_actions.
                                                        The probabilities must sum to 1.0 for each action. The index of the probabilities corresponds to the index 
                                                        of the actions in used_actions. If None, uniform distribution is assumed. 
    """

    def __init__(
            self,
            env:MiniGridEnv,
            used_actions: list[Actions] = None,
            prob_distribution: dict[Actions,list[float]]= None
            ):
        
        if used_actions is not None and prob_distribution is not None:
            assert len(used_actions) == len(prob_distribution), "Length of used_actions must match length of prob_distribution"
            for l in prob_distribution.values():
                assert abs(sum(l)-1.0) < 1e-6 
        

        self.env = env
        self.used_actions = used_actions if used_actions is not None else [action for action in Actions]
        self.prob_distribution = prob_distribution if prob_distribution is not None else {action: [1/len(self.used_actions) for _ in self.used_actions] for action in self.used_actions}
        
        #keep track of added lava positions. Used in reset() to restore the lava, so that it stays at the same positions. 
        self.lava_pos:tuple[int,int] | None = None
        #reset initial env such that env.agent_pos and other variables are initialized
        self.env.reset() 
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, 'metadata', {})
        
    def step(self, action: ActType):
        # Convert integer action to Actions enum
        if isinstance(action, (int, np.integer)):
            action_enum = Actions(action)
        else:
            action_enum = action
            
        # Choose action index based on probabilities, then get the actual action
        action_idx = np.random.choice(len(self.used_actions), p=self.prob_distribution[action_enum])
        actual_action = self.used_actions[action_idx]
        
        # Convert back to integer for the underlying MiniGrid environment
        return self.env.step(actual_action.value)
    
    def reset(self, seed:int | None = None, options=None): 
        obs, info = self.env.reset(seed=seed, options=options)
        
        #restore lava if it was added before
        if self.lava_pos is not None:
            self.env.grid.set(self.lava_pos[0], self.lava_pos[1], Lava())
            
        return obs, info

    def __getattr__(self, name):
        # Delegate all other attributes/methods to the wrapped env
        return getattr(self.env, name) 

    def convert_to_probabilistic_storm(self):
        #We store states as hashes of envs using the MiniGridEnv.hash function, which returns the same hashes for deepcopies of same envs. When storing states as envs, the bird api will use the __hash__ and __eq__ functions of the env, which do not have the desired properties for deepcopies.
        init = self.env.hash() 
        #We use the dict to get an env from a hash in the delta function. This way, we can simulate steps by using the defined step() functions. 
        visited_envs = {init: self.env}

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
            return cell_type

        def available_actions(s: str):
            """
            Up until now it is assumed that all actions are always available. 
            TODO: Change this if needed.
            """
            # We reformat the used_action list to the format expected by the bird api
            return [[action.name] for action in self.used_actions]

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
            probs = self.prob_distribution[given_action]
            for i,action in enumerate(self.used_actions): 
                if action == Actions.pickup: 
                    fwd_pos = self.env.front_pos
                    fwd_cell = curr_env.grid.get(fwd_pos[0], fwd_pos[1])
                    #in case of pickup, chech if the cell in front can actually be picked up. 
                    #If not, then we know that the minigrid code will not change the state, so we can directly return the current state with the corresponding probability. 
                    if not fwd_cell or not fwd_cell.can_pickup():
                        result.append((probs[i], s))
                        continue
                elif action == Actions.toggle: 
                    fwd_pos = self.env.front_pos
                    fwd_cell = curr_env.grid.get(fwd_pos[0], fwd_pos[1])
                    #in case of pickup, chech if the cell in front can actually be picked up. 
                    #If not, then we know that the minigrid code will not change the state, so we can directly return the current state with the corresponding probability. 
                    if not fwd_cell: #There is the function toggle() in WorldObj, but it then exectues side effects on the env, which we do not want here. Hence, we ony skip empty cells.
                        result.append((probs[i], s))
                        continue

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
                        modeltype=ModelType.MDP,
                        max_size=100000
                        )
        
        # Create hash-to-state-ID mapping for efficient lookup during simulations
        # The visited_envs dict preserves the order in which states were discovered,
        # which corresponds directly to the StormVogel state IDs (0, 1, 2, ...)
        self.hash_to_state_id = {}
        for state_id, env_hash in enumerate(visited_envs.keys()):
            self.hash_to_state_id[env_hash] = state_id
        self.model = model
    
            
        return model, visited_envs

    def get_current_state_id(self) -> int:
        """
        Get the StormVogel state ID corresponding to the current environment state.
        
        Returns:
            int: The StormVogel state ID for the current environment state.
            
        Raises:
            ValueError: If the current environment state was not encountered during conversion.
        """
        if not hasattr(self, 'hash_to_state_id'):
            raise ValueError("Environment has not been converted to StormVogel model yet. Call convert_to_probabilistic_storm() first.")
        
        current_hash = self.env.hash()
        if current_hash not in self.hash_to_state_id:
            raise ValueError(f"Current environment state (hash: {current_hash}) was not encountered during StormVogel conversion.")
            
        return self.hash_to_state_id[current_hash]

    def add_lava(self, pos: tuple = None) -> tuple[int,int] | None:
        """
        Args: 
            pos (tuple[int,int], optional): Specific position to place lava. If None, a random position is chosen (only tiles not occupied by another WorldObject will be considered). Defaults to None.
        
        Returns: 
            tuple[int,int] | None: The position where lava was placed, or None if no lava was placed.
        """
        
        # Check if there are already Lava objects in the grid
        if any(isinstance(obj, Lava) for obj in self.env.grid.grid if obj is not None):
            print("Environment already contains lava hazards.")
            return None
        if pos is not None:
            self.env.grid.set(pos[0], pos[1], Lava())
            print(f"Placed lava at specified position ({pos[0]}, {pos[1]})")
            return pos
        
        valid_positions = []
        # Check all positions in the grid (excluding outer walls) for not being occupied or starting pos of agent
        for x in range(1, self.env.grid.width - 1):
            for y in range(1, self.env.grid.height - 1):
                if self._is_valid_lava_position(x, y):
                    valid_positions.append((x, y))

        if not valid_positions:
            print("No valid positions found for lava placement.Specify pos to override other objects.")
            return None
        
        # Randomly choose one position
        chosen_pos = self.env.np_random.choice(len(valid_positions))
        (x, y) = valid_positions[chosen_pos]
        self.env.grid.set(x, y, Lava())
        self.lava_pos = (x,y)  # Store the lava position for restoration on reset
        print(f"Placed lava at position ({x}, {y})")    
        return (x,y)
    
    def _is_valid_lava_position(self, x, y):
        """Check if a position is safe for placing lava (won't make environment unsolvable)."""
        # Position must be empty
        if self.env.grid.get(x, y) is not None:
            return False 
        # Don't place at agent starting position
        if (x, y) == tuple(self.env.agent_pos):
            return False
        return True