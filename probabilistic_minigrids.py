from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ActType
from minigrid.core.actions import Actions
from minigrid.core.world_object import Lava
from stormvogel import bird, ModelType
import copy


class ProbabilisticEnvWrapper:
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
                assert abs(sum(l)-1.0) < 1e-6 #Probabilities for each action must sum to 1.0
        # Additional initialization for probabilistic elements can be added here
        

        self.env = env
        self.used_actions = used_actions if used_actions is not None else [action for action in Actions]
        self.prob_distribution = prob_distribution if prob_distribution is not None else {action: [1/len(self.used_actions) for _ in self.used_actions] for action in self.used_actions}
        
    def step(self, action: ActType):
        actual_action = self.env.np_random.choice(self.used_actions, p=self.prob_distribution[action])
        return self.env.step(actual_action)

    def __getattr__(self, name):
        # Delegate all other attributes/methods to the wrapped env
        return getattr(self.env, name) 

    def convert_to_probabilistic_storm(self):
        self.reset()

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
        return model, visited_envs
