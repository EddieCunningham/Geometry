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
from src.instances.manifolds import *
from src.instances.vector_fields import *
import src.util as util

def run_all():
  jax.config.update("jax_enable_x64", True)
  rng_key = random.PRNGKey(0)

  # Get a vector field
  lie_G = SpaceOfMatrices(dim=4)
  G = lie_G.G
  TeG = TangentSpace(G.e, G)
  v = random.normal(rng_key, (16,))
  V = TangentVector(v, TeG)
  X = lie_G.left_invariant_vector_field(V)

  # Construct its integral curve
  rng_key, key = random.split(rng_key, 2)
  p0 = random.normal(key, (4, 4))
  p = p0
  gamma = IntegralCurve(p0, V=X)

  # Try evaluating the map
  pt = gamma(0.4)

  # Some maps for later
  def _F(x, inverse=False):
    if inverse == False:
      return 2*x + 2.0
    return 0.5*(x - 2.0)
  F = Diffeomorphism(_F, domain=G, image=G)

  _f = lambda x: jnp.linalg.norm(jnp.sin(jnp.arange(16).reshape(4, 4)@(3 + x)**2).ravel())
  f = Map(_f, domain=G, image=Reals())

  # Create a flow
  theta = Flow(G, X)

  # Check that the group law theta_t + theta_s = theta_{t+s}
  t, s = 0.2, 0.4
  out1 = compose(theta.get_theta_t(t), theta.get_theta_t(s))(pt)
  out2 = compose(theta.get_theta_t(s), theta.get_theta_t(t))(pt)
  theta_p = theta.get_theta_p(pt)
  out3 = theta_p(s + t)
  assert jnp.allclose(out1, out3)
  assert jnp.allclose(out2, out3)
  assert jnp.allclose(theta_p(0.0), pt)

  # Check diffeomorphism invariance: pushfwd(F, X) is flow of F(theta_t(F^{-1}(.)))
  FX = pushforward(F, X)
  flow = Flow(G, FX)

  theta_t = theta.get_theta_t(t)
  check = compose(F, theta_t, F.get_inverse())

  rhs = flow(t, p)
  lhs = check(p)
  assert jnp.allclose(lhs, rhs)

  # import pdb; pdb.set_trace()

if __name__ == "__main__":
  from debug import *
  # jax.config.update('jax_disable_jit', True)
  jax.config.update("jax_enable_x64", True)
  run_all()