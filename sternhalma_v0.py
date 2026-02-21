from sternhalma.env.sternhalma import env, raw_env
from sternhalma.env.rllib_wrapper import DiscreteActionMaskWrapper


def rllib_env(max_actions: int = 256, **kwargs):
    return DiscreteActionMaskWrapper(env(**kwargs), max_actions=max_actions)


__all__ = ["env", "raw_env", "rllib_env", "DiscreteActionMaskWrapper"]
