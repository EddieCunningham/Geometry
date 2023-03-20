from functools import partial
from typing import Callable, Optional
import src.util
import jax.random as random
import jax.numpy as jnp
from src.set import *
from src.map import *
from src.manifold import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util

def get_random_point(M: Manifold, rng_key: Optional["random.PRNGKey"]=None):
  """Initialize a manifold
  """
  if rng_key is None:
    rng_key = random.PRNGKey(0)

  dim = M.dimension
  x_coords = random.normal(rng_key, (dim,))

  # Get a random point from the manifold on each chart
  for i, c in enumerate(M.atlas.charts):
    p = c.inverse(x_coords)
    assert p in M

  return p

def EuclideanManifold_test():
  M = EuclideanManifold()
  p = get_random_point(M)


################################################################################################################

def run_all():

  # Construct the different manifolds and call their member functions
  m1 = EuclideanManifold(dimension=4)
  m2 = GeneralLinearGroup(dim=4)
  m3 = Sphere(dim=4)
  m4 = RealProjective(dim=4)

  rng_key = random.PRNGKey(3)

  # for i, m in enumerate([m1, m2, m4]):
  for i, m in enumerate([m1, m2, m3, m4]):
    get_random_point(m, rng_key)

if __name__ == "__main__":
  from debug import *
