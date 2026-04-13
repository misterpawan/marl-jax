"""Common MARL environment wrapper classes."""

from marl.wrappers.autoreset_wrapper import AutoResetWrapper
from marl.wrappers.hierarchy_wrapper import HierarchyVecWrapper
from marl.wrappers.merge_marl_wrapper import MergeWrapper
from marl.wrappers.observation_action_wrapper import \
    ObservationActionRewardWrapper
from marl.wrappers.overcooked import OverCooked
from marl.wrappers.ssd import SSDWrapper

__all__ = [
    "AutoResetWrapper",
    "HierarchyVecWrapper",
    "MergeWrapper",
    "ObservationActionRewardWrapper",
    "OverCooked",
    "SSDWrapper",
    "MeltingPotWrapper",
]


def __getattr__(name):
  if name == "MeltingPotWrapper":
    from marl.wrappers.meltingpot_wrapper import MeltingPotWrapper
    return MeltingPotWrapper
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
