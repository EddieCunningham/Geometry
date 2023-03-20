import jax
import jax.tree_util

__all__ = ["tree_shapes",
           "soft_assert",
           "GLOBAL_CHECK"]

GLOBAL_CHECK = False

def tree_shapes(x):
  return jax.tree_util.tree_map(lambda x: x.shape, x)

def soft_assert(x):
  assert x
  pass
