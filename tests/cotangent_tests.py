from functools import partial
from typing import Callable
from typing import NewType
import src.util
import jax
import jax.numpy as jnp
import jax.random as random
from src.set import *
from src.map import *
from src.tangent import *
from src.cotangent import *
from src.manifold import *
from src.instances.manifolds import *
import nux
import src.util as util

def liebniz_test(v: CotangentVector, f: Function, g: Function):
  p = v.coTpM.p

  fg = Map(lambda x: f(x)*g(x), domain=f.domain, image=f.image)

  product_rule = f(p)*v(g) + g(p)*v(f)
  ans = v(fg)
  assert jnp.allclose(product_rule, ans)

################################################################################################################

def get_test_tangent_vector(M: Manifold, p: Point, rng_key):
  assert p in M

  # Construct the tangent space at p
  coTpM = CotangentSpace(p, M)

  # Get a tangent vector
  x_coords = random.normal(rng_key, (M.dimension,))
  v = CotangentVector(x_coords, coTpM)
  return v

def tangent_test(M: Manifold, p: Point, f: Function, g: Function, rng_key):
  assert f.domain == M
  assert g.domain == M

  # Get a tangent vector and apply it to a function
  k1, k2 = random.split(rng_key, 2)
  v1 = get_test_tangent_vector(M, p, k1)
  v2 = get_test_tangent_vector(M, p, k2)

  # Construction
  out = v1(f)

  # Liebniz rule
  liebniz_test(v1, f, g)

  # Test that we can add vectors together correctly
  v1f = v1(f)
  v2f = v2(f)

  v1pv2 = v1 + v2
  v1pv2f = v1pv2(f)

  assert jnp.allclose(v1pv2f, v1f + v2f)

################################################################################################################

def differential_test(M: Manifold, N: Manifold, p: Point, F: Map, G: Map, f: Function, g: Function, rng_key):
  assert p in M
  assert F.domain == M
  assert F.image == N
  assert f.domain == N
  assert g.domain == N

  # Construct the differential of F at p
  dFp = Differential(F, p)

  # Get a tangent vector
  v = get_test_tangent_vector(M, p, rng_key)

  # Apply it to a vector
  dFp_v = dFp(v)

  # Test Liebniz rule
  liebniz_test(dFp_v, f, g)

  # Composition test
  q = F(p)
  dGq = Differential(G, q)
  dGq_dFp_v = dGq(dFp_v)

  dGFp = Differential(compose(G, F), p)
  dGFp_v = dGFp(v)

  # Make sure that all of the coordinates line up
  dFp_coords = dFp.get_coordinates()
  dGq_coords = dGq.get_coordinates()
  dGFp_coords = dGFp.get_coordinates()
  assert jnp.allclose(dGFp_coords, dGq_coords@dFp_coords)

  assert jnp.allclose(dGFp_v.x, dGq_dFp_v.x)

################################################################################################################

def run_all():
  from tests.manifold_tests import get_random_point
  # jax.config.update('jax_disable_jit', True)

  rng_key = random.PRNGKey(0)

  M = EuclideanManifold(dimension=4)
  N = EuclideanManifold(dimension=3)
  p = get_random_point(M)

  f = Map(lambda x: jnp.linalg.norm(x), domain=M, image=Reals())
  g = Map(lambda x: jax.nn.softmax(x).sum(), domain=M, image=Reals())
  tangent_test(M, p, f, g, rng_key)

  rng_key = random.PRNGKey(0)
  def Fx(x):
    matrix = random.normal(rng_key, ((3, 4)))
    return matrix@x
  F = Map(Fx, domain=M, image=N)

  def Gx(x):
    matrix = random.normal(rng_key, ((3, 3)))
    return matrix@x
  G = Map(Gx, domain=N, image=N)

  f = Map(lambda x: jnp.linalg.norm(x), domain=N, image=Reals())
  g = Map(lambda x: jax.nn.softmax(x).sum(), domain=N, image=Reals())


  differential_test(M, N, p, F, G, f, g, rng_key)

  # Test that the tangent bundle works as well
  TM = CotangentBundle(M)

  # The projection map should be a submersion
  coTpM = CotangentSpace(p, M)
  x_coords = jnp.arange(M.dimension) + 1.0
  v = CotangentVector(x_coords, coTpM)

  assert TM.projection_map().is_submersion((p, v))

  # Try with something less trivial
  M = Sphere(dim=3)
  N = Sphere(dim=3)
  p = p/jnp.linalg.norm(p)
  f = Map(lambda x: jnp.linalg.norm(x), domain=M, image=Reals())
  g = Map(lambda x: jax.nn.softmax(x).sum(), domain=M, image=Reals())
  tangent_test(M, p, f, g, rng_key)

  def on_sphere(x, inverse=False):
    if inverse == False:
      rphi = nux.CartesianToSpherical()(x[None])[0]
      r, phi = rphi[...,:1], rphi[...,1:]
      phi = phi + 0.1
      rphi = jnp.concatenate([r, phi], axis=-1)
      return nux.SphericalToCartesian()(rphi)[0][0]
    else:
      rphi = nux.CartesianToSpherical()(x[None])[0]
      r, phi = rphi[...,:1], rphi[...,1:]
      phi = phi - 0.1
      rphi = jnp.concatenate([r, phi], axis=-1)
      return nux.SphericalToCartesian()(rphi)[0][0]

  F = Diffeomorphism(on_sphere, domain=M, image=M)
  differential_test(M, M, p, F, F, f, g, rng_key)


if __name__ == "__main__":
  from debug import *
  # run_all()