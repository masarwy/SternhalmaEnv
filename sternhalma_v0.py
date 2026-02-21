from sternhalma.env.sternhalma import env, raw_env
from sternhalma.env.discrete_action_wrapper import DiscreteActionMaskWrapper


def discrete_action_env(max_actions: int = 256, **kwargs):
    return DiscreteActionMaskWrapper(env(**kwargs), max_actions=max_actions)


__all__ = ["env", "raw_env", "discrete_action_env", "DiscreteActionMaskWrapper"]
