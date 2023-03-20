from functools import partial
from typing import Callable, Optional
import src.util
import jax.random as random
import nux
import jax
import jax.numpy as jnp
from src.set import *
from src.map import *
from src.manifold import *
from src.tangent import *
from src.lie_group import *
from src.lie_algebra import *
from src.vector_field import *
from src.flow import *
from src.bundle import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util

################################################################################################################

def run_all():
  from tests.manifold_tests import get_random_point
  # jax.config.update('jax_disable_jit', True)

  rng_key = random.PRNGKey(0)

  # Get a point on the manifold
  M = EuclideanGroup(dim=4)
  bA, tangent_coords = random.normal(rng_key, (2, 4, 5))
  tangent_coords = tangent_coords.ravel()
  b, A = bA[:,-1], bA[:,:-1]
  p = (b, A)

  # # Get a point on the manifold
  # M = GeneralLinearGroup(dim=4)
  # A, tangent_coords = random.normal(rng_key, (2, 4, 4))
  # tangent_coords = tangent_coords.ravel()
  # p = A

  # Create the tangent bundle
  TM = TangentBundle(M)

  # Create a local trivialization so that we can get a tangent space
  UxF = TM.local_trivialization(p)
  TpM = UxF.fiber

  # Get a tangent vector
  v = TangentVector(tangent_coords, TpM)

  # Construct a point on the bundle
  x = (p, v)

  # Check that the projection map works
  pi = TM.get_projection_map()
  out = pi(x)
  assert all(jax.tree_util.tree_map(jnp.allclose, out, p))

  # Try getting the differential of the projection map
  dpi = pi.get_differential(x)

  # Apply it to a tangent vector of the local trivialization
  local_trivialization_tangent_coords = random.normal(rng_key, (2*20,))
  TxE = TangentSpace(x, UxF)
  vx = TangentVector(local_trivialization_tangent_coords, TxE)

  # Ensure that the differential mapped the tangent vector to the correct place
  check = dpi(vx)
  assert check.TpM.manifold == M

  # Check that pi is a submersion
  assert pi.is_submersion(x)

  assert 0, "Need to check bundle homomorphisms!"

if __name__ == "__main__":
  from debug import *
  run_all()