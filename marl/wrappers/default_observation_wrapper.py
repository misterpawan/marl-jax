"""
The Default Observation Wrapper adds an empty field to the observation.
For instance, this wrapper can be used to add an INVENTORY observation for all
players, if one is not already there. This is useful to match APIs between
different environments that expose different observations.

Ref: https://github.com/deepmind/meltingpot/blob/85131b6c6dba2c48caf0e56334ef74bc70f65962/meltingpot/python/utils/scenarios/wrappers/default_observation_wrapper.py
"""

from collections.abc import Mapping
from typing import Any, Optional

import dm_env
from meltingpot.python.utils.substrates import substrate
import numpy as np


def _setdefault(dictionary: Mapping[str, Any], key: str,
                value: Any) -> Mapping[str, Any]:
  """Sets the default value of `key` to `value` if necessary.
  Args:
    dictionary: the dictionary to add a default for.
    key: The key to add a default for.
    value: The default value to add if key is missing.
  Returns:
    Either dictionary or a copy with the default value added.
  """
  if key in dictionary:
    return dictionary
  else:
    return dict(dictionary, **{key: value})


class Wrapper(substrate.Substrate):
  """Wrapper to add observations with default values if not actually present."""

  def __init__(self,
               env: substrate.Substrate,
               key: str,
               default_value: np.ndarray,
               default_spec: Optional[dm_env.specs.Array] = None):
    """Initializer.
    Args:
      env: environment to wrap. When this wrapper closes env will also be
        closed.
      key: field name of the observation to add.
      default_value: The default value to add to the observation.
      default_spec: The default spec for the observation to add. By default,
        this will be set to match default_value. If specified this must match
        the default value.
    """
    super().__init__(env)
    self._key = key
    self._default_value = default_value.copy()
    self._default_value.flags.writeable = False
    if default_spec is None:
      self._default_spec = dm_env.specs.Array(
          shape=self._default_value.shape,
          dtype=self._default_value.dtype,
          name=self._key)
    else:
      self._default_spec = default_spec
    self._default_spec.validate(self._default_value)

  def reset(self):
    """See base class."""
    timestep = super().reset()
    observation = [
        _setdefault(obs, self._key, self._default_value)
        for obs in timestep.observation
    ]
    return timestep._replace(observation=observation)

  def step(self, action):
    """See base class."""
    timestep = super().step(action)
    observation = [
        _setdefault(obs, self._key, self._default_value)
        for obs in timestep.observation
    ]
    return timestep._replace(observation=observation)

  def observation_spec(self):
    """See base class."""
    observation_spec = super().observation_spec()
    return [
        _setdefault(obs, self._key, self._default_spec)
        for obs in observation_spec
    ]
