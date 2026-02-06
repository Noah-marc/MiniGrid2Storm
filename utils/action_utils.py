"""
Action conversion utilities for Minigrid2Storm project.

This module provides conversion functions between different action representations:
- minigrid.Actions (canonical format - the source of truth)
- stormvogel.model.Action (for formal verification)
- gym integers (for RL training)
- bird.Action format (list[str] for model construction)

The canonical format throughout the project is minigrid.Actions enum.
All conversions are stateless pure functions.
"""

from minigrid.core.actions import Actions
from stormvogel.model import Action as StormvogelAction


# ============================================================================
# Conversions TO canonical format (minigrid.Actions)
# ============================================================================

def from_stormvogel_action(action: StormvogelAction) -> Actions:
    """
    Convert StormVogel Action to MiniGrid Actions enum.
    
    Args:
        action: StormVogel Action with labels as frozenset of strings
        
    Returns:
        MiniGrid Actions enum corresponding to the action name
        
    Example:
        >>> sv_action = Action(frozenset({"forward"}))
        >>> from_stormvogel_action(sv_action)
        <Actions.forward: 2>
    """
    action_name = next(iter(action.labels))
    return Actions[action_name]


def from_gym_int(action_int: int) -> Actions:
    """
    Convert gym integer action to MiniGrid Actions enum.
    
    Args:
        action_int: Integer representing the action (from gym action space)
        
    Returns:
        MiniGrid Actions enum
        
    Example:
        >>> from_gym_int(2)
        <Actions.forward: 2>
    """
    return Actions(action_int)


# ============================================================================
# Conversions FROM canonical format (minigrid.Actions)
# ============================================================================

def to_stormvogel_action(action: Actions) -> StormvogelAction:
    """
    Convert MiniGrid Actions enum to StormVogel Action format.
    
    Used when passing actions to shields or formal verification tools.
    
    Args:
        action: MiniGrid Actions enum
        
    Returns:
        StormVogel Action with the action name as a frozen set label
        
    Example:
        >>> to_stormvogel_action(Actions.forward)
        Action(labels=frozenset({'forward'}))
    """
    return StormvogelAction(frozenset({action.name}))


def to_gym_int(action: Actions) -> int:
    """
    Convert MiniGrid Actions enum to gym integer format.
    
    Used when interfacing with gym environments or RL libraries.
    
    Args:
        action: MiniGrid Actions enum
        
    Returns:
        Integer value of the action
        
    Example:
        >>> to_gym_int(Actions.forward)
        2
    """
    return action.value


def to_bird_action(action: Actions) -> list[str]:
    """
    Convert MiniGrid Actions enum to bird Action format.
    
    Bird API expects actions as list of strings. This is primarily used
    during model construction with the stormvogel.bird API.
    
    Args:
        action: MiniGrid Actions enum
        
    Returns:
        List containing the action name as a single string
        
    Example:
        >>> to_bird_action(Actions.forward)
        ['forward']
    """
    return [action.name]


# ============================================================================
# Convenience functions for batch conversions
# ============================================================================

def actions_to_bird_format(actions: list[Actions]) -> list[list[str]]:
    """
    Convert a list of MiniGrid Actions to bird format.
    
    Useful for the available_actions() function in model construction.
    
    Args:
        actions: List of MiniGrid Actions enums
        
    Returns:
        List of actions in bird format (list of list of strings)
        
    Example:
        >>> actions_to_bird_format([Actions.left, Actions.right, Actions.forward])
        [['left'], ['right'], ['forward']]
    """
    return [to_bird_action(action) for action in actions]
