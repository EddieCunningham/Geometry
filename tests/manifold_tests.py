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

def test_euclidean_manifold():
  rng_key = random.PRNGKey(0)
  M = EuclideanManifold(dimension=4)
  p = random.normal(rng_key, shape=(M.dimension,))

  # Check that the point is in the manifold
  assert p in M

  # Get a chart and get coordinates
  chart = M.get_chart_for_point(p)
  x = chart(p)

def test_general_linear_group():
  rng_key = random.PRNGKey(0)
  M = GLRn(dim=4)
  p = random.normal(rng_key, shape=(M.N, M.N))

  # Check that the point is in the manifold
  assert p in M

  # Get a chart and get coordinates
  chart = M.get_chart_for_point(p)
  x = chart(p)

def test_sphere():
  rng_key = random.PRNGKey(0)
  M = Sphere(dim=4)
  p = random.normal(rng_key, shape=(M.dimension + 1,)); p = p / jnp.linalg.norm(p)

  # Check that the point is in the manifold
  assert p in M

  # Get a chart and get coordinates
  chart = M.get_chart_for_point(p)
  x = chart(p)

def test_real_projective():
  rng_key = random.PRNGKey(0)
  M = RealProjective(dim=4)
  p = random.normal(rng_key, shape=(M.dimension + 1,))

  # Check that the point is in the manifold
  assert p in M

  # Get a chart and get coordinates
  chart = M.get_chart_for_point(p)
  x = chart(p)

def test_cartesian_product():
  rng_key = random.PRNGKey(0)
  k1, k2 = random.split(rng_key, 2)

  N = Sphere(dim=4)
  G1 = GLRn(dim=4)
  G2 = GLRn(dim=4)
  M = CartesianProductManifold(N, G1, G2)

  q = random.normal(rng_key, shape=(N.dimension + 1,)); q = q / jnp.linalg.norm(q)
  g1, g2 = random.normal(k2, shape=(2, G1.N, G1.N))
  p = (q, g1, g2)

  # Check that the point is in the manifold
  assert p in M

  # Get a chart and get coordinates
  chart = M.get_chart_for_point(p)
  x = chart(p)

################################################################################################################

def run_all():
  test_euclidean_manifold()
  test_general_linear_group()
  test_sphere()
  test_real_projective()
  test_cartesian_product()

if __name__ == "__main__":
  from debug import *
  run_all()