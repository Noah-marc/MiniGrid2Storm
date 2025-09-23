from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.lavagap import LavaGapEnv
from gymnasium.core import ActType
from minigrid.core.actions import Actions    

class ProbabilisticEnvWrapper:
    def __init__(
            self,
            env:MiniGridEnv,
            used_actions: list[Actions] = None,
            prob_distribution: list[float]= None):
        
        if used_actions is not None and prob_distribution is not None:
            assert len(used_actions) == len(prob_distribution), "Length of used_actions must match length of prob_distribution"
            assert abs(sum(prob_distribution) - 1.0) < 1e-6, "Probabilities in prob_distribution must sum to 1.0"
        # Additional initialization for probabilistic elements can be added here
        
        self.env = env
        self.used_actions = used_actions if used_actions is not None else [action for action in Actions]
        self.prob_distribution = prob_distribution

    def step(self, action: ActType):
        actual_action = self.env.np_random.choice(self.used_actions, p=self.prob_distribution)
        return self.env.step(actual_action)

    def __getattr__(self, name):
        # Delegate all other attributes/methods to the wrapped env
        return getattr(self.env, name) 
