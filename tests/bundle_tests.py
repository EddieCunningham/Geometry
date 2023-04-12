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
from src.section import *
from src.vector_field import *
from src.flow import *
from src.bundle import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util
from tests.cotangent_tests import get_chart_fun

################################################################################################################

def tangent_bundle_tests():
  jax.config.update("jax_enable_x64", True)
  from tests.manifold_tests import get_random_point
  # jax.config.update('jax_disable_jit', True)

  rng_key = random.PRNGKey(0)

  # Get a point on the manifold
  M = EuclideanGroup(dim=4)
  bA, tangent_coords = random.normal(rng_key, (2, 4, 5))
  tangent_coords = tangent_coords.ravel()
  b, A = bA[:,-1], bA[:,:-1]
  A, _ = jnp.linalg.qr(A)
  p = (b, A)

  # Get a tangent vector, which is a point on the bundle
  TpM = TangentSpace(p, M)
  v = TangentVector(tangent_coords, TpM)

  # Create the tangent bundle
  TM = TangentBundle(M)

  # Check that the projection map works
  pi = TM.get_projection_map()
  out = pi(v)
  assert all(jax.tree_util.tree_map(jnp.allclose, out, p))

  # Check that the local trivialization map works
  Phi = TM.get_local_trivialization_map(v)
  out = Phi(v)

  # Try getting the differential of the projection map
  dpi = pi.get_differential(v)

  # Apply it to a tangent vector of the local trivialization
  local_trivialization_tangent_coords = random.normal(rng_key, (40,))

  TxE = TangentSpace(v, TM)
  vx = TangentVector(local_trivialization_tangent_coords, TxE)

  # Ensure that the differential mapped the tangent vector to the correct place
  check = dpi(vx)
  assert check.TpM.manifold == M

  # Check that pi is a submersion
  assert pi.is_submersion(v)

  # The global differential is a smooth bundle homomorphism covering F
  def _F(p, inverse=False):
    b, A = p
    Q, _ = jnp.linalg.qr(random.normal(rng_key, shape=A.shape))
    if inverse == False:
      b *= 2
      A = A@Q
    else:
      b /= 2
      A = A@Q.T
    return (b, A)

  # Construct a bundle homomorphsim
  F = InvertibleMap(_F, domain=M, image=M)
  dF = GlobalDifferential(F, M)

  # Test it out
  out = dF(v)

  # Check that the induced map on sections is linear over functions
  from tests.vector_field_tests import get_vector_field_fun
  from src.instances.vector_fields import AutonomousVectorField
  k1, k2 = random.split(rng_key, 2)

  # Construct vector fields so that they output the correct shapes
  s1 = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  s2 = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)

  def F_tilde(s):
    # The global differential is our bundle homomorphism
    return apply_bundle_homomorphism_to_section(dF, s)

  def _f1(p):
    b, A = p
    return (b**2).sum() + (jnp.cos(A)).sum()

  def _f2(p):
    b, A = p
    return (jnp.sin(b)).sum() + (A**2).sum()

  f1 = Map(_f1, domain=M, image=Reals())
  f2 = Map(_f2, domain=M, image=Reals())

  out1 = (F_tilde(f1*s1 + f2*s2))(p)
  out2 = (f1*F_tilde(s1) + f2*F_tilde(s2))(p)

  # Check that they're the same
  assert jnp.allclose(out1.x, out2.x)

################################################################################################################

def frame_bundle_tests():
  jax.config.update("jax_enable_x64", True)
  from tests.manifold_tests import get_random_point
  # jax.config.update('jax_disable_jit', True)

  rng_key = random.PRNGKey(0)

  # Get a point on the manifold
  M = EuclideanGroup(dim=4)
  bA, *tangent_coords = random.normal(rng_key, (21, 4, 5))
  tangent_coords = [x.ravel() for x in tangent_coords]
  b, A = bA[:,-1], bA[:,:-1]
  A, _ = jnp.linalg.qr(A)
  p = (b, A)

  # Create a basis for the tangent space
  TpM = TangentSpace(p, M)
  basis = []
  for coords in tangent_coords:
    v = TangentVector(coords, TpM)
    basis.append(v)

  basis = TangentBasis(basis, TpM)

  # Create the frame bundle
  FB = FrameBundle(M)

  # Check that the projection map works
  pi = FB.get_projection_map()
  out = pi(basis)
  assert all(jax.tree_util.tree_map(jnp.allclose, out, p))

  # Check that the local trivialization map works
  Phi = FB.get_local_trivialization_map(basis)
  out = Phi(basis)

  # Check that pi is a submersion
  assert pi.is_submersion(basis)

  # Construct a frame bundle homomorphism by applying action of GL(n,R)
  class GLWithAction(GLRn):

    def right_action_map(self, g: Point, FB: FrameBundle) -> Map[TangentBasis,TangentBasis]:
      def theta_g(basis: TangentBasis) -> TangentBasis:
        lt_map = FB.get_local_trivialization_map(basis)
        p, mat = lt_map(basis)
        new_mat = mat@g
        return lt_map.inverse((p, new_mat))
      return Map(theta_g, domain=FB, image=FB)

  gl_with_action = GLWithAction(dim=M.dimension)
  g = random.normal(rng_key, (M.dimension, M.dimension))
  bundle_homomorphism = gl_with_action.right_action_map(g, FB)

  # Test it out
  out = bundle_homomorphism(basis)

  # Check that the induced map on sections is linear over functions
  from tests.vector_field_tests import get_vector_field_fun
  from src.instances.vector_fields import AutonomousFrame
  k1, k2 = random.split(rng_key, 2)

  # Construct vector fields so that they output the correct shapes
  s1 = AutonomousFrame(get_chart_fun(M.dimension, k1), M)
  s2 = AutonomousFrame(get_vector_field_fun(M.dimension, k2), M)

  def F_tilde(s):
    # The global differential is our bundle homomorphism
    return apply_bundle_homomorphism_to_section(bundle_homomorphism, s)

  def _f1(p):
    b, A = p
    return (b**2).sum() + (jnp.cos(A)).sum()

  def _f2(p):
    b, A = p
    return (jnp.sin(b)).sum() + (A**2).sum()

  f1 = Map(_f1, domain=M, image=Reals())
  f2 = Map(_f2, domain=M, image=Reals())

  out1 = (F_tilde(f1*s1 + f2*s2))(p)
  out2 = (f1*F_tilde(s1) + f2*F_tilde(s2))(p)

  # Check that they're all the same
  assert sum([not jnp.allclose(a.x, b.x) for a, b in zip(out1.basis, out2.basis)]) == 0

  # Check that we can split the frame bundle into a list of vector fields
  elements = s1.to_vector_field_list()

  s1p = s1(p)
  for i, Xp in enumerate(s1p):
    Yp = elements[i](p)
    assert jnp.allclose(Xp.x, Yp.x)

  # Check that we can get the corresponding dual coframes to frames
  s1_dual = s1.get_dual_coframe()
  s1_dualp = s1_dual(p)

  # Check that the covectors and vectors are compatable
  for i, Ep in enumerate(s1p):
    for j, ep in enumerate(s1_dualp):
      out = ep(Ep)
      assert jnp.allclose(out, i==j)

  s1_reconstr = s1_dual.get_dual_frame()
  s1p_reconstr = s1_reconstr(p)
  for i, (Xp, Yp) in enumerate(zip(s1p, s1p_reconstr)):
    assert jnp.allclose(Xp.x, Yp.x)


def run_all():
  tangent_bundle_tests()
  frame_bundle_tests()

if __name__ == "__main__":
  from debug import *
  run_all()