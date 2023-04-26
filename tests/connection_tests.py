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
from src.bundle import *
from src.flow import *
from src.connection import *
import nux
import src.util as util
from tests.vector_field_tests import get_vector_field_fun
from tests.tensor_tests import get_tensor_field_fun
from tests.cotangent_tests import get_chart_fun

def lie_algebra_valued_alternating_tensor_tests():
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  dimension = 4
  M = GLRn(dim=dimension)
  p = random.normal(rng_key, (M.N, M.N))
  TpM = TangentSpace(p, M)

  # Create a vector space that we'll use
  V = M.lieG

  # Create some alternating tensors that we'll use to construct a vector
  # valued alternating tensor
  def create_alternating_tensor(rng_key, k):
    tensor_type = TensorType(0, k)
    TkTpM = TensorSpace(p, tensor_type, M)
    coords = random.normal(rng_key, ([M.dimension]*sum(tensor_type)))
    T = Tensor(coords, TkTpM=TkTpM)
    return make_alternating(T)

  tensor_type_k = 3
  rng_key, *keys = random.split(rng_key, 1 + V.dimension)
  ws = [create_alternating_tensor(key, tensor_type_k) for key in keys]

  # Create a vector alternating tensor
  w = LieAlgebraValuedAlternatingTensor(ws, basis=V.get_basis())

  # Evaluate it on some tangent vectors
  def create_tangent_vector(rng_key):
    coords = random.normal(rng_key, (M.dimension,))
    return TangentVector(coords, TpM)

  rng_key, *keys = random.split(rng_key, 1 + tensor_type_k)
  Xs = [create_tangent_vector(key) for key in keys]
  X1, X2, X3 = Xs

  # Check that the tensor is alternating
  out1 = w(X1, X2, X3)
  out2 = -w(X1, X3, X2)
  out3 = -w(X2, X1, X3)
  out4 = w(X2, X3, X1)
  out5 = w(X3, X1, X2)
  out6 = -w(X3, X2, X1)
  assert jnp.allclose(out1(p).x, out2(p).x)
  assert jnp.allclose(out1(p).x, out3(p).x)
  assert jnp.allclose(out1(p).x, out4(p).x)
  assert jnp.allclose(out1(p).x, out5(p).x)
  assert jnp.allclose(out1(p).x, out6(p).x)

################################################################################################################

def lie_algebra_valued_differential_form_tests():
  rng_key = random.PRNGKey(0)

  # Now try the same thing but on lie valued differential forms
  dim = 3
  M = GLRn(dim=dim)
  lieG = M.lieG
  p, q = random.normal(rng_key, (2, dim, dim))
  tensor_type_k = 2

  tensor_type = TensorType(0, tensor_type_k)
  def make_tf(rng_key):
    return get_tensor_field_fun(M, tensor_type, rng_key, input_dimension=dim**2)

  keys = random.split(rng_key, lieG.dimension)
  tfs = [make_tf(key) for key in keys]
  w = ParametricLieAlgebraValuedDifferentialForm(tfs, tensor_type, M, lieG)

  def create_vector_field(rng_key):
    return AutonomousVectorField(get_vector_field_fun(M.dimension, rng_key), M)
  rng_key, *keys = random.split(rng_key, 2 + tensor_type_k)
  Xs = [create_vector_field(key) for key in keys]
  X1, X2, X3 = Xs

  # Check that the tensor is alternating
  out1 = w(X1, X2)
  out2 = -w(X2, X1)
  assert jnp.allclose(out1(p)(q).x, out2(p)(q).x)

  # Try doing an exterior derivative
  dw = exterior_derivative(w)

  # check = dw(p)
  # X1p = X1(p)
  # X2p = X2(p)
  # X3p = X3(p)
  # check(X1p, X2p, X3p)
  # import pdb; pdb.set_trace()

  out = dw(X1, X2, X3)
  test1 = out(p)
  # test2 = test1(q)
  import pdb; pdb.set_trace()

################################################################################################################

def maurer_cartan_form_test():
  rng_key = random.PRNGKey(0)
  k1, k2, k3, k4, k5, k6 = random.split(rng_key, 6)

  # Construct a manifold
  dimension = 4
  G = GLRn(dim=dimension)
  lieG = G.lieG
  g = random.normal(k1, (G.N, G.N))
  h = random.normal(k2, (G.N, G.N))

  # # Check that we can get a basis and dual basis for the Lie algebra
  # basis_vector_fields = lieG.get_basis()
  # dual_basis_covector_fields = lieG.get_dual_basis()
  # for i, Ei in enumerate(basis_vector_fields):
  #   for j, thetaj in enumerate(dual_basis_covector_fields):
  #     term = thetaj(Ei)
  #     out = term(g)
  #     if i == j:
  #       assert jnp.allclose(out, 1.0)
  #     else:
  #       assert jnp.allclose(out, 0.0)

  # Construct a left invariant vector field
  coord = random.normal(k3, (G.dimension,))
  Ae = TangentVector(coord, lieG.TeG)
  A = lieG.get_left_invariant_vector_field(Ae)

  # Construct the maurer cartan form
  theta = maurer_cartan_form(G)

  # Check that it leaves left invariant vector fields unchanged
  # A_comp: G -> Lie(G).  However, it doesn't matter which input
  # point in G that we use because they will all map to the same
  # element of the Lie algebra in this case.
  A_comp = theta(A)

  lhs = A(h)
  rhs = A_comp(G.e)(h)

  assert jnp.allclose(lhs.x, rhs.x)

  # Check the exterior derivative

  import pdb; pdb.set_trace()

################################################################################################################

def connection_tests():
  rng_key = random.PRNGKey(0)
  k1, k2, k3, k4, k5, k6 = random.split(rng_key, 6)

  # Construct a manifold
  dimension = 4
  M = GLRn(dim=dimension)
  p = random.normal(k1, (M.N, M.N))

  G = GLRn(dim=dimension)
  lieG = G.lieG
  g = random.normal(k2, (G.N, G.N))
  h = random.normal(k3, (G.N, G.N))
  ginv = jnp.linalg.inv(g)

  # Get a principal G bundle
  class MatrixGBundle(PrincipalGBundle, ProductBundle):
    def get_action_map(self, right=True):
      def R(ug):
        u, g = ug
        p, h = u
        return (p, h@g)
      domain = CartesianProductManifold(self, self.G)
      return Map(R, domain=domain, image=self)

    def get_filled_right_action_map(self, g: Point):
      g_inv = jnp.linalg.inv(g)
      def Rg(u, inverse=False):
        p, h = u
        return (p, h@g) if inverse == False else (p, h@g_inv)
      return Diffeomorphism(Rg, domain=self, image=self)

  P = MatrixGBundle(M, G)
  u = (p, h)

  # Get the fundamental vector field
  # Get a tangent vector on TgG
  coord = random.normal(k3, (G.dimension,))
  Ae = TangentVector(coord, lieG.TeG)
  A = lieG.get_left_invariant_vector_field(Ae)

  Astar = P.fundamental_vector_field(A)
  out = Astar(u)

  # R: ((p, g), h) -> (p, g@h)
  R = P.get_action_map(right=True)

  # Rg: (p, g) -> (p, g@h)
  Rg = P.get_filled_right_action_map(g)
  out1 = R((u, g))
  out2 = Rg(u)
  assert jnp.allclose(out1[0], out2[0])
  assert jnp.allclose(out1[1], out2[1])

  # The fundamental vector field of (Rg)*A equals
  # that of (Ag(ginv)A)
  lhs = pushforward(Rg, Astar)
  Ad = G.get_adjoint_representation()
  Ad_ginv = Ad(ginv)
  rhs = P.fundamental_vector_field(Ad_ginv(A))

  out1 = lhs(u)
  out2 = rhs(u)
  assert jnp.allclose(out1.x, out2.x)

  # Maurer cartan form tests:
  w0 = maurer_cartan_form(G)
  A_comp = w0(A)

  lhs = A(h)
  rhs = A_comp(h)

  import pdb; pdb.set_trace()

################################################################################################################

def run_all():
  # jax.config.update('jax_disable_jit', True)
  jax.config.update("jax_enable_x64", True)

  # lie_algebra_valued_alternating_tensor_tests()
  lie_algebra_valued_differential_form_tests()
  # maurer_cartan_form_test()
  # connection_tests()

  # connection_tests_frame()

if __name__ == "__main__":
  from debug import *

  run_all()
