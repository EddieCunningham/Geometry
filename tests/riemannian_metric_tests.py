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
from src.vector_field import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
from src.instances.vector_fields import *
from src.section import *
from src.tensor import *
from src.lie_derivative import *
from src.riemannian_metric import *
import nux
import src.util as util
from tests.vector_field_tests import get_vector_field_fun
from tests.tensor_tests import get_tensor_field_fun

################################################################################################################

def example_13p11():
  # Construct the manifolds
  M = EuclideanManifold(dimension=2)
  N = EuclideanManifold(dimension=3)

  # Get a pair of vector fields and a point to use
  rng_key = random.PRNGKey(0)
  k1, k2, k3 = random.split(rng_key, 3)
  X1 = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  X2 = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)
  p = random.normal(k3, (M.dimension,))

  # Construct the map
  def _F(uv):
    u, v = uv
    return jnp.array([u*jnp.cos(v), u*jnp.sin(v), v])
  F = Map(_F, domain=M, image=N)

  # Pull the Euclidean metric back through the map
  g_ = EuclideanMetric(dimension=N.dimension)
  g = pullback_tensor_field(F, g_)

  # Get the coordinate function differentials of M
  u_mask = jnp.zeros(M.dimension, dtype=bool)
  u_mask = u_mask.at[0].set(True)
  u = M.atlas.charts[0].get_slice_chart(mask=u_mask)

  v_mask = jnp.zeros(M.dimension, dtype=bool)
  v_mask = v_mask.at[1].set(True)
  v = M.atlas.charts[0].get_slice_chart(mask=v_mask)

  # Helper for function differential
  d = lambda f: as_tensor_field(FunctionDifferential(f))

  # Get the differentials
  du = d(u)
  dv = d(v)

  # Get du^2 and dv^2
  du2 = tensor_field_product(du, du)
  dv2 = tensor_field_product(dv, dv)

  # Compute the analytical solution
  g_comp = du2 + (u*u + 1.0)*dv2

  test1 = g(X1, X2)(p)
  test2 = g_comp(X1, X2)(p)

  assert jnp.allclose(test1, test2)

  test1 = g(p)(X1(p), X2(p))
  test2 = g(p)(X1(p), X2(p))
  assert jnp.allclose(test1, test2)

################################################################################################################

def gradient_tests():
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=4)
  p = random.normal(rng_key, (5,)); p = p/jnp.linalg.norm(p)

  # Create a Riemannian metric
  tf = get_tensor_field_fun(M, TensorType(0, 2), rng_key)
  g = ParametricRiemannianMetric(tf, M)

  # Create some vector fields
  k1, k2, k3 = random.split(rng_key, 3)
  X = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  Y = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)

  # Check that the musical isomorphisms work
  flat = TangentToCotangetBundleIsomorphism(g)
  sharp = flat.get_inverse()
  X_sharp = flat(X)
  X_reconstr = sharp(X_sharp)

  assert jnp.allclose(X(p).x, X_reconstr(p).x)

  # Test the gradient
  f = Map(lambda x: jnp.cos(x).sum(), domain=M, image=Reals())
  grad_f = gradient_vector_field(f, g)

  test1 = flat(grad_f)(X)
  test2 = X(f)

  assert jnp.allclose(test1(p), test2(p))

################################################################################################################

def run_all():
  jax.config.update("jax_enable_x64", True)
  example_13p11()
  gradient_tests()

if __name__ == "__main__":
  from debug import *
  run_all()