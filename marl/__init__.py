"""MARL package initialization and JAX compatibility shims."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp


def _ensure_jax_compat():
  # Older Acme versions still reference deprecated JAX symbols that were removed
  # in newer 0.4.x releases. Restore lightweight aliases so the repo can use a
  # GPU-capable modern JAX build without patching site-packages.
  if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map
  if not hasattr(jax, "tree_multimap"):
    jax.tree_multimap = jax.tree_util.tree_map
  if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array
  if not hasattr(jax, "xla"):
    jax.xla = SimpleNamespace(Device=jax.Device, DeviceArray=jax.Array)
  elif not hasattr(jax.xla, "Device"):
    jax.xla.Device = jax.Device
  if not hasattr(jax.xla, "DeviceArray"):
    jax.xla.DeviceArray = jax.Array
  if not hasattr(jax, "pxla"):
    jax.pxla = SimpleNamespace(ShardedDeviceArray=jax.Array)
  elif not hasattr(jax.pxla, "ShardedDeviceArray"):
    jax.pxla.ShardedDeviceArray = jax.Array
  if not hasattr(jnp, "DeviceArray"):
    jnp.DeviceArray = jax.Array


_ensure_jax_compat()
