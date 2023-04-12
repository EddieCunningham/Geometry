from functools import partial
from typing import Callable
from typing import NewType
import itertools
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
from src.differential_form import *
from src.lie_group import *
from src.lie_algebra import *
from src.flow import *
import nux
import src.util as util
from tests.vector_field_tests import get_vector_field_fun
from tests.tensor_tests import get_tensor_field_fun
from tests.cotangent_tests import get_chart_fun


def one_parameter_subgroup_tests():

  # Create a Lie algebra
  dim = 4
  G = GLRn(dim=dim)
  lieG = G.get_lie_algebra()

  # Construct an element of the Lie algebra
  rng_key = random.PRNGKey(0)
  k1, k2, k3 = random.split(rng_key, 3)
  A_coords = random.normal(rng_key, (dim**2,))
  Ae = TangentVector(A_coords, lieG.TeG)

  t = 0.3
  s = -0.5

  # Show that the one parameter subgroups for the general linear group are matrix exponentials
  A = lieG.get_left_invariant_vector_field(Ae)
  ops = lieG.get_one_parameter_subgroup(A)

  # Check that the one parameter subgroup is the same as the matrix exponenital
  out1 = ops(t)
  out2 = jax.scipy.linalg.expm(t*A_coords.reshape((dim, dim)))
  assert jnp.allclose(out1, out2)

  # # Get the exponential map and try using it
  exp = lieG.get_exponential_map()
  out3 = exp(t*A)
  assert jnp.allclose(out1, out3)

  # Proposition 20.8
  # b)
  out1 = exp((s + t)*A)
  out2 = G.multiplication_map(exp(s*A), exp(t*A))
  assert jnp.allclose(out1, out2)

  # c)
  eA = exp(A)
  out1 = G.inverse(eA)
  out2 = exp(-A)
  assert jnp.allclose(out1, out2)

  # d)
  out1 = G.multiplication_map(eA, eA)
  out2 = exp(2.0*A)
  assert jnp.allclose(out1, out2)

  # e)
  _zero = TangentVector(jnp.zeros_like(A_coords), lieG.TeG)
  zero = lieG.get_left_invariant_vector_field(_zero)

  # Get the differential at 0
  dexp_0 = exp.get_differential(zero)
  coords = dexp_0.get_coordinates()
  assert jnp.allclose(coords, jnp.eye(coords.shape[0]))

  # g) If H is another Lie group and Phi: G -> H is a Lie group
  # homomorphism, then exp(Phi(.)) = Phi_star(exp(.))
  # Conjugation is a Lie group homomorphism
  g = random.normal(k1, (dim, dim))
  Phi = G.conjugation_map(g)

  # Get the induced Lie algebra homomorphism for Phi
  out1 = exp(induced_lie_algebra_homomorphism(Phi, A))
  out2 = Phi(exp(A))
  assert jnp.allclose(out1, out2)

  # h) Flow of left invariant vector field is R_{exptX}
  theta = Flow(G, lieG.get_left_invariant_vector_field(Ae))
  out1 = theta(t, g)
  out2 = G.right_translation_map(exp(t*A))(g)
  assert jnp.allclose(out1, out2)

################################################################################################################

def infinitesmal_generator_tests():
  dim = 4
  rng_key = random.PRNGKey(0)
  k1, k2, k3, k4, k5 = random.split(rng_key, 5)

  # Create a manifold
  M = GLRn(dim=dim)
  p = random.normal(k2, (dim, dim))

  # Create a Lie group with a right action on M
  class GroupOverFrame(GLRn):

    def right_orbit_map(self, p: Point, M: Manifold) -> Map[Point,Point]:
      def theta_p(g):
        assert p.shape == (dim, dim)
        assert g.shape == (dim, dim)
        return p@g
      return Map(theta_p, domain=self, image=M)

    def left_orbit_map(self, p: Point, M: Manifold) -> Map[Point,Point]:
      def theta_p(g):
        assert p.shape == (dim, dim)
        assert g.shape == (dim, dim)
        return g@p
      return Map(theta_p, domain=self, image=M)

  G = GroupOverFrame(dim=dim)
  lieG = G.get_lie_algebra()

  # Get an element of the Lie algebra of G
  X_coords = random.normal(k2, (dim**2,))
  Xe = TangentVector(X_coords, lieG.TeG)
  X = lieG.get_left_invariant_vector_field(Xe)

  # Create the flow over M induced by the group action
  flow = FlowInducedByGroupAction(M, G, X)
  t, s = 0.3, -0.1

  # Check that it is a flow
  assert jnp.allclose(flow(0.0, p), p)
  assert jnp.allclose(flow(t, flow(s, p)), flow(t + s, p))

  # Check the infinitesmal generator
  def flow_p(t):
    return flow(t, p)
  _, Vp = jax.jvp(flow_p, (0.0,), (1.0,))

  Xhat_p = flow.infinitesmal_generator(p)

  # Get an element of the Lie algebra of G
  Y_coords = random.normal(k3, (dim**2,))
  Ye = TangentVector(Y_coords, lieG.TeG)
  Y = lieG.get_left_invariant_vector_field(Ye)

  # Theorem 20.15: Infinitesmal generator is a Lie algebra homomorphism
  theta_hat = get_infinitesmal_generator_map(G, M)
  test1 = theta_hat(lieG.bracket(X, Y))
  test2 = lie_bracket(theta_hat(X), theta_hat(Y))
  assert jnp.allclose(test1(p).x, test2(p).x)

  # Check that the generator using the left action is an antihomomorphism
  theta_hat = get_infinitesmal_generator_map(G, M, right_action=False)
  test1 = theta_hat(lieG.bracket(X, Y))
  test2 = lie_bracket(theta_hat(X), theta_hat(Y))
  assert jnp.allclose(test1(p).x, -test2(p).x)

################################################################################################################

def adjoint_representation_tests():
  dim = 4
  rng_key = random.PRNGKey(0)
  k1, k2, k3, k4, k5 = random.split(rng_key, 5)

  # Create a manifold
  M = GLRn(dim=dim)
  p = random.normal(k2, (dim, dim))

  # Create a Lie group with a right action on M
  class GroupOverFrame(GLRn):

    def right_orbit_map(self, p: Point, M: Manifold) -> Map[Point,Point]:
      def theta_p(g):
        assert p.shape == (dim, dim)
        assert g.shape == (dim, dim)
        return p@g
      return Map(theta_p, domain=self, image=M)

    def left_orbit_map(self, p: Point, M: Manifold) -> Map[Point,Point]:
      def theta_p(g):
        assert p.shape == (dim, dim)
        assert g.shape == (dim, dim)
        return g@p
      return Map(theta_p, domain=self, image=M)

  G = GroupOverFrame(dim=dim)
  lieG = G.get_lie_algebra()
  g, g1, g2 = random.normal(k1, (3, dim, dim))

  # Get an element of the Lie algebra of G
  X_coords, Y_coords = random.normal(k2, (2, dim**2,))
  Xe = TangentVector(X_coords, lieG.TeG)
  X = lieG.get_left_invariant_vector_field(Xe)

  Ye = TangentVector(Y_coords, lieG.TeG)
  Y = lieG.get_left_invariant_vector_field(Ye)


  # Check the adoint representation
  Ad = G.get_adjoint_representation()

  # Run some inputs through it
  check = Ad(g)(X)(p)

  # Check composition
  test1 = Ad(G.multiplication_map(g1, g2))
  test2 = compose(Ad(g1), Ad(g2))
  out1 = test1(X)(p)
  out2 = test2(X)(p)
  assert jnp.allclose(out1.x, out2.x)

  # Inverse
  test1 = compose(Ad(g), Ad(g).get_inverse())
  out1 = test1(X)
  assert jnp.allclose(out1(p).x, X(p).x)

  # Adjoint representation of Lie algebra
  ad = lieG.get_adjoint_representation()
  adX = ad(X)

  # Check that it is the pushforward of Ad
  # Ad_starX = induced_lie_algebra_homomorphism(Ad, X)

  test1 = adX(Y)(p)

  # TODO: FIX THIS
  # test2 = Ad_starX(Y)(p)
  # assert jnp.allclose(test1.x, test2.x)
  # import pdb; pdb.set_trace()

################################################################################################################

def run_all():
  # jax.config.update('jax_disable_jit', True)
  jax.config.update("jax_enable_x64", True)

  one_parameter_subgroup_tests()
  infinitesmal_generator_tests()
  adjoint_representation_tests()

if __name__ == "__main__":
  from debug import *

  util.GLOBAL_CHECK = False
  run_all()
  util.GLOBAL_CHECK = True
