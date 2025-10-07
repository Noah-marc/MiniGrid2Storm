from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.lavagap import LavaGapEnv
from gymnasium.core import ActType
from minigrid.core.actions import Actions    

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
