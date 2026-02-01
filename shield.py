
from stormvogel.stormpy_utils.model_checking import model_checking
from minigrid.core.actions import Actions
from stormvogel.model import Action, Model

class Shield: 

    def verify_action(self, state: int, action: Action) -> Actions:
        """ Should be overwritten by child classes.
            The function should implement the shielding logic and either return the iven action if allowed or give an alternative action
            It is set up such that you can just pass actions from the minigrid Actions enum. If you use it with Stormvogel or another library, make sure to convert properly.
        """
        return action
    
class DeltaShield(Shield): 

    def __init__(self, model: Model, safety_property: str, delta:float =0.5): 
        
        self.model = model
        self.safety_property = safety_property
        self.delta = delta
        self.min_probs = model_checking(model, safety_property)
        self.optimal_safety_policy = self.min_probs.scheduler
        

        self.blocked = 0
        self.not_blocked = 0

    def _action_value(self, state:int, action:Action): 
        """ Compute the value of taking action in state."""

        choice_for_state = self.model.choices[state]        
        if action not in choice_for_state.transition:
            action_labels = list(action.labels) if hasattr(action, 'labels') else str(action)
            raise ValueError(f"Action {action_labels} not found in state {state}")
        
        subranches = choice_for_state.transition[action]
        value = 0.0

        for prob_s_prime, s_prime in subranches.branch: 
            min_prob_s_prime = self.min_probs.values[s_prime.id]
            value += prob_s_prime * min_prob_s_prime
        return value

    
    def verify_action(self, state: int, action: Actions) -> Actions: 
        """
        Implements delta-shielding logic.
            The function checks if the given action is allowed in the given state according to delta-shielding.
            If allowed, it returns the given action; otherwise, it blocks the action (returns None).
        """
        assert state is not None
        assert action is not None

        act_val = self._action_value(state, action)
        optimal_act_val = self.min_probs.values[state]
        if self.delta * act_val <= optimal_act_val:
            self.not_blocked += 1
            return action
        else:
            self.blocked += 1 
            return self.optimal_safety_policy.get_choice_of_state(state)


