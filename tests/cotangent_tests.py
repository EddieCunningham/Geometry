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
from src.section import *
from src.lie_derivative import *
import nux
import src.util as util
from tests.vector_field_tests import get_vector_field_fun
from src.instances.vector_fields import AutonomousVectorField, AutonomousCovectorField

################################################################################################################

def get_chart_fun(dimension, rng_key):
  # Construct the vector field over M
  import nux
  flow = nux.GLOW(n_layers=2,
           working_dim=16,
           hidden_dim=32,
           n_resnet_layers=2)

  x = random.normal(rng_key, (100, dimension))
  flow(x, rng_key=rng_key)
  params = flow.get_params()

  def apply_fun(x, inverse=False):
    return flow(x[None], params=params, rng_key=rng_key, inverse=inverse)[0][0]

  return apply_fun

################################################################################################################

def run_all():
  jax.config.update("jax_enable_x64", True)
  rng_key = random.PRNGKey(0)

  # Check that we can get coordinates correctly
  chart = get_chart_fun(dimension=6, rng_key=rng_key)
  Rn = EuclideanManifold(dimension=6, chart=chart)

  p = random.normal(rng_key, (6,))
  TpRn = TangentSpace(p, EuclideanManifold(dimension=6))

  W = AutonomousCovectorField(get_vector_field_fun(Rn.dimension, rng_key), Rn)
  w = W(p)

  basis_vectors = TpRn.get_basis()
  out = [w(v) for v in basis_vectors]
  Id = Chart(lambda x, inverse=False: x, domain=Rn, image=Reals(Rn.dimension))
  check = w.get_coordinates(Id)


  # Construct a manifold
  M = Sphere(dim=4)
  p = random.normal(rng_key, (5,)); p = p/jnp.linalg.norm(p)

  # Construct a basis for the tangent space
  tangent_coords = random.normal(rng_key, (M.dim, M.dim))
  TpM = TangentSpace(p, M)
  basis = []
  for coords in tangent_coords:
    v = TangentVector(coords, TpM)
    basis.append(v)
  tangent_basis = TangentBasis(basis, TpM)

  # Get the dual basis
  cotangent_basis = CotangentBasis.from_tangent_basis(tangent_basis)

  # Check that the cotangent vectors pair with the tangent vectors
  for i, v in enumerate(tangent_basis.basis):
    for j, w in enumerate(cotangent_basis.basis):
      out = w(v)
      assert jnp.allclose(out, i==j)

  # Test cotangent vector fields
  key1, key2, key3 = random.split(rng_key, 3)
  X = AutonomousVectorField(get_vector_field_fun(M.dimension, key1), M)
  W = AutonomousCovectorField(get_vector_field_fun(M.dimension, key2), M)
  v = X(p)
  w = W(p)

  # Make sure that we can apply vector fields to covector fields
  assert jnp.allclose((W(X))(p), w(v))

  ############################################################
  ############################################################
  # Check the pullback map


  # Build a map to matrices
  def _F(p):
    out = random.normal(rng_key, (5, 5))*jnp.sin(p)
    return out
  N = GeneralLinearGroup(dim=5)
  F = Map(_F, domain=M, image=N)
  Fp = F(p)

  # Get a vector field over the output
  W = AutonomousCovectorField(get_vector_field_fun(N.dimension, key2), N)
  w = W(Fp)

  out1 = F.get_pullback(p)(w)(v)
  out2 = w(F.get_differential(p)(v))

  assert jnp.allclose(out1, out2)

  # Check that we can pullback covector fields
  F_W = pullback(F, W)
  out3 = F_W(p)(v)

  assert jnp.allclose(out1, out3)

  # Test some other identities
  def _u(p):
    out = random.normal(rng_key, (5, 5))@jnp.sin(p)
    return out.sum()
  u = Map(_u, domain=N, image=EuclideanManifold(dimension=1))

  # Check that the derivation to get Proposition 11.25 checks out
  check1 = pullback(F, u*W)(p)
  check2 = Pullback(F, p)(u(Fp)*W(Fp))
  assert jnp.allclose(check1.x, check2.x)
  check3 = u(Fp)*Pullback(F, p)(W(Fp))
  assert jnp.allclose(check1.x, check3.x)
  check4 = u(Fp)*pullback(F, W)(p)
  assert jnp.allclose(check1.x, check4.x)
  check5 = (compose(u, F)*pullback(F, W))(p)
  assert jnp.allclose(check1.x, check5.x)

  # Test the differential of a scalar function
  du = FunctionDifferential(u, N)
  out1 = pullback(F, du)(p)
  out2 = FunctionDifferential(compose(u, F), M)(p)

  assert jnp.allclose(out1.x, out2.x)

  ############################################################
  # Proposition 11.45
  key1, key2, key3 = random.split(rng_key, 3)
  X = AutonomousVectorField(get_vector_field_fun(M.dimension, key1), M)
  Y = AutonomousVectorField(get_vector_field_fun(M.dimension, key2), M)

  # Need a closed covector field
  u = Map(_u, domain=M, image=EuclideanManifold(dimension=1))
  W = FunctionDifferential(u, M)

  t1 = X(W(Y))
  t2 = Y(W(X))
  t3 = W(lie_bracket(X, Y))

  assert jnp.allclose(t1(p) - t2(p), t3(p))

if __name__ == "__main__":
  from debug import *
  run_all()