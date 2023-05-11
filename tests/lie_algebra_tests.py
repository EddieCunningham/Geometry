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
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util

def run_all():
  jax.config.update("jax_enable_x64", True)
  # Test the Lie bracket identities
  G = GLRn(dim=4)
  lie_G = G.get_lie_algebra()

  # Tangent space at identity
  TeG = TangentSpace(G.e, G)

  # Start with some tangent vector
  rng_key = random.PRNGKey(0)
  k1, k2 = random.split(rng_key, 2)

  # Some scalars and a function for later
  a, b = random.normal(rng_key, (2,))
  _f = lambda x: jnp.linalg.norm(jnp.sin(jnp.arange(16).reshape(4, 4)@(3 + x)**2).ravel())
  f = Map(_f, domain=G, image=EuclideanManifold(dimension=1))

  v1, v2, v3 = random.normal(k1, (3, 16))
  V1 = TangentVector(v1, TeG)
  V2 = TangentVector(v2, TeG)
  V3 = TangentVector(v3, TeG)

  # Make left invariant vector fields
  X = lie_G.get_left_invariant_vector_field(V1)
  Y = lie_G.get_left_invariant_vector_field(V2)
  Z = lie_G.get_left_invariant_vector_field(V3)

  # Ensure that X and Y are actually left invariant
  # This means that (L_*)X = X
  g, h = random.normal(k2, (2, 4, 4))
  L_X = pushforward(G.left_translation_map(g), X)
  lhs = L_X(h)(f)
  rhs = X(h)(f)
  assert jnp.allclose(lhs, rhs)

  # Bilinearity
  p = random.normal(rng_key, (4, 4))
  lhs = lie_G.bracket(a*X + b*Y, Z)(p)(f)
  rhs = (a*lie_G.bracket(X, Z) + b*lie_G.bracket(Y, Z))(p)(f)
  assert jnp.allclose(lhs, rhs)

  # Antisymmetry
  lhs = lie_G.bracket(X, Y)(p)(f)
  rhs = -lie_G.bracket(Y, X)(p)(f)
  assert jnp.allclose(lhs, rhs)

  # Jacobi Identity
  t1 = lie_G.bracket(X, lie_G.bracket(Y, Z))
  t2 = lie_G.bracket(Y, lie_G.bracket(Z, X))
  t3 = lie_G.bracket(Z, lie_G.bracket(X, Y))
  check = (t1 + t2 + t3)(p)(f)
  assert jnp.allclose(check, 0.0)

  # import pdb; pdb.set_trace()

if __name__ == "__main__":
  from debug import *
  # jax.config.update('jax_disable_jit', True)
  jax.config.update("jax_enable_x64", True)
  run_all()