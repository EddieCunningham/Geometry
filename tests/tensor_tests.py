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
import nux
import src.util as util
from tests.vector_field_tests import get_vector_field_fun
from tests.cotangent_tests import get_chart_fun

################################################################################################################

def cotangent_tensor_product_test():
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=4)
  p = random.normal(rng_key, (M.dimension + 1,)); p = p/jnp.linalg.norm(p)

  # First build a tensor
  tensor_type = TensorType(3, 3)
  tensor_coords = random.normal(rng_key, [4]*sum(tensor_type))
  TkTpM = TensorSpace(p, tensor_type, M)
  T = Tensor(tensor_coords, TkTpM=TkTpM)

  # Evaluate it on some cotangent and tangent vectors
  coords = random.normal(rng_key, (sum(tensor_type), M.dim))
  coTpM = CotangentSpace(p, M)
  ws = [CotangentVector(x, coTpM) for x in coords[:tensor_type.k]]

  TpM = TangentSpace(p, M)
  vs = [TangentVector(x, TpM) for x in coords[tensor_type.k:]]
  test = T(*ws, *vs)

  # Check that we can change coordinates correctly
  # Do this by creating a manifold which uses a different set of charts
  class SphereCustomChart(Sphere):

    def get_atlas(self):
      atlas = super().get_atlas()

      # Create a diffeomorphism that we can compose with the regular chart
      _phi = get_chart_fun(dimension=self.dimension, rng_key=rng_key)
      phi = Diffeomorphism(_phi, domain=EuclideanManifold(dimension=self.dimension), image=EuclideanManifold(dimension=self.dimension))

      new_charts = []
      for chart in atlas.charts:
        new_chart = compose(phi, chart)
        new_charts.append(new_chart)

      return Atlas(new_charts)

  M2 = SphereCustomChart(dim=4)

  # Create the same tensor as before, but using the
  # new coordinates
  tensor_coords2 = T.get_coordinates(M2.get_chart_for_point(p))
  TkTpM2 = TensorSpace(p, tensor_type, M2)
  T2 = Tensor(tensor_coords2, TkTpM=TkTpM2)

  # Evaluate the two tensors on the same inputs to
  test2 = T2(*ws, *vs)
  assert jnp.allclose(test, test2)

  # Now try a tensor product
  rng_key, _ = random.split(rng_key, 2)
  tangent_coords = random.normal(rng_key, (3, M.dim))
  TpM = TangentSpace(p, M)
  v1 = TangentVector(tangent_coords[0], TpM)
  v2 = TangentVector(tangent_coords[1], TpM)
  v3 = TangentVector(tangent_coords[2], TpM)

  w1, w2, w3 = ws[:3]
  w1w2 = tensor_product(w1, w2)
  out = w1w2(v1, v2)

  # Try a more complicated tensor product
  w1w2w3 = tensor_product(tensor_product(w1, w2), w3)
  out2 = w1w2w3(v1, v2, v3)

  # Check associativity
  w1w2w3_2 = tensor_product(w1, tensor_product(w2, w3))
  out2_2 = w1w2w3_2(v1, v2, v3)
  assert jnp.allclose(out2, out2_2)

  # Check symmetrization
  SymT = symmetrize(w1w2w3)
  out1 = SymT(v1, v2, v3)
  out2 = SymT(v1, v3, v2)
  out3 = SymT(v2, v1, v3)
  out4 = SymT(v2, v3, v1)
  out5 = SymT(v3, v1, v2)
  out6 = SymT(v3, v2, v1)
  assert jnp.allclose(out1, out2)
  assert jnp.allclose(out1, out3)
  assert jnp.allclose(out1, out4)
  assert jnp.allclose(out1, out5)
  assert jnp.allclose(out1, out6)

  out1_ = SymT.call_unvectorized(v1, v2, v3)
  out2_ = SymT.call_unvectorized(v1, v3, v2)
  out3_ = SymT.call_unvectorized(v2, v1, v3)
  out4_ = SymT.call_unvectorized(v2, v3, v1)
  out5_ = SymT.call_unvectorized(v3, v1, v2)
  out6_ = SymT.call_unvectorized(v3, v2, v1)
  assert jnp.allclose(out1_, out1)
  assert jnp.allclose(out2_, out2)
  assert jnp.allclose(out3_, out3)
  assert jnp.allclose(out4_, out4)
  assert jnp.allclose(out5_, out5)
  assert jnp.allclose(out6_, out6)

  ##################################################

  # Next try the tensor product of two tensors
  tensor_type_1, tensor_type_2 = TensorType(1, 2), TensorType(2, 2)
  tensor_coords_1 = random.normal(rng_key, [4]*sum(tensor_type_1))
  tensor_coords_2 = random.normal(rng_key, [4]*sum(tensor_type_2))
  TkTpM_1 = TensorSpace(p, tensor_type_1, M)
  TkTpM_2 = TensorSpace(p, tensor_type_2, M)

  T1 = Tensor(tensor_coords_1, TkTpM=TkTpM_1)
  T2 = Tensor(tensor_coords_2, TkTpM=TkTpM_2)

  T1T2 = tensor_product(T1, T2)

  # Evaluate it on some cotangent and tangent vectors
  coords = random.normal(rng_key, (sum(T1T2.type), M.dim))
  coTpM = CotangentSpace(p, M)
  ws = [CotangentVector(x, coTpM) for x in coords[:T1T2.type.k]]

  TpM = TangentSpace(p, M)
  vs = [TangentVector(x, TpM) for x in coords[T1T2.type.k:]]
  out1 = T1T2(*ws, *vs)
  out2 = T1T2.call_unvectorized(*ws, *vs)
  assert jnp.allclose(out1, out2)


################################################################################################################

def get_tensor_field_fun(M, tensor_type, rng_key):
  manifold_dimension = M.dimension
  dimension = manifold_dimension**(tensor_type.k + tensor_type.l)
  # Construct the tensor field over M
  import nux
  net = nux.CouplingResNet1D(out_dim=dimension,
                             working_dim=8,
                             hidden_dim=16,
                             nonlinearity=nux.util.square_swish,
                             dropout_prob=0.0,
                             n_layers=1)
  x = random.normal(rng_key, (100, manifold_dimension))
  net(x, rng_key=rng_key)
  params = net.get_params()
  leaves, treedef = jax.tree_util.tree_flatten(params)
  keys = random.split(rng_key, len(leaves))
  new_leaves = [random.normal(key, x.shape) for x, key in zip(leaves, keys)]
  params = treedef.unflatten(new_leaves)

  def vf(x):
    return net(x[None], params=params, rng_key=rng_key)[0].reshape([manifold_dimension]*(tensor_type.k + tensor_type.l))

  return vf

################################################################################################################

def tensor_field_tests():
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=4)
  p = random.normal(rng_key, (5,)); p = p/jnp.linalg.norm(p)

  # Construct some tensor fields
  k1, k2, k3 = random.split(rng_key, 3)
  tensor_type = TensorType(3, 3)
  T = AutonomousTensorField(get_tensor_field_fun(M, tensor_type, k1), tensor_type, M)

  # Try evaluating the fields
  Tp = T(p)

  # Try applying a covariant tensor field to vector fields
  tensor_type = TensorType(0, 3)
  T = AutonomousTensorField(get_tensor_field_fun(M, tensor_type, k2), tensor_type, M)
  Tp = T(p)

  k1, k2, k3, k4 = random.split(k3, 4)
  X1 = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  X2 = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)
  X3 = AutonomousVectorField(get_vector_field_fun(M.dimension, k3), M)
  X4 = AutonomousVectorField(get_vector_field_fun(M.dimension, k4), M)
  test = T(X1, X2, X3)(p)

  X1p = X1(p)
  X2p = X2(p)
  X3p = X3(p)
  X4p = X4(p)

  # Apply a function to the tensor
  f = Map(lambda x: x.sum(), domain=M, image=Reals())
  out1 = (f*T)(p)(X1(p), X2(p), X3(p))
  out2 = (f(p))*(T(p)(X1(p), X2(p), X3(p)))
  assert jnp.allclose(out1, out2)

  # Check that the tensor field is multilinear over functions
  f2 = Map(lambda x: jnp.linalg.norm(jnp.sin(x)), domain=M, image=Reals())
  out1 = T(X1, f*X2 + f2*X4, X3)(p)
  out2 = f(p)*T(X1, X2, X3)(p) + f2(p)*T(X1, X4, X3)(p)
  assert jnp.allclose(out1, out2)

  # Check that we can do a tensor product of tensor fields
  tensor_type = TensorType(1, 2)
  T1 = AutonomousTensorField(get_tensor_field_fun(M, tensor_type, k1), tensor_type, M)

  tensor_type = TensorType(2, 1)
  T2 = AutonomousTensorField(get_tensor_field_fun(M, tensor_type, k2), tensor_type, M)

  w1 = AutonomousCovectorField(get_vector_field_fun(M.dimension, k4), M)
  w2 = AutonomousCovectorField(get_vector_field_fun(M.dimension, k2), M)
  w3 = AutonomousCovectorField(get_vector_field_fun(M.dimension, k1), M)

  w1p = w1(p)
  w2p = w2(p)
  w3p = w3(p)

  T1T2 = tensor_field_product(T1, T2)

  out1 = T1T2(p)(w1p, w2p, w3p, X1p, X2p, X3p)
  out2 = T1T2(w1, w2, w3, X1, X2, X3)(p)
  assert jnp.allclose(out1, out2)


  # Check that the pullbacks work

  # Build a map to matrices
  def _F(p):
    out = random.normal(rng_key, (5, 5))*jnp.sin(p)
    return out
  N = GeneralLinearGroup(dim=5)
  F = Map(_F, domain=M, image=N)
  Fp = F(p)

  # Make a tensor field product to test
  tensor_type = TensorType(0, 1)
  T1 = AutonomousTensorField(get_tensor_field_fun(N, tensor_type, k2), tensor_type, N)
  tensor_type = TensorType(0, 2)
  T2 = AutonomousTensorField(get_tensor_field_fun(N, tensor_type, k3), tensor_type, N)

  # tensor_type = TensorType(0, 1)
  # T3 = AutonomousTensorField(get_tensor_field_fun(N, tensor_type, k1), tensor_type, N)
  T = tensor_field_product(T1, T2)

  # tensor_type = TensorType(0, 3)
  # T = AutonomousTensorField(get_tensor_field_fun(N, tensor_type, k3), tensor_type, N)

  f = Map(lambda x: x.sum(), domain=N, image=Reals())

  Tp = T(Fp)

  dFp = F.get_differential(p)
  out1 = Tp(dFp(X1p), dFp(X2p), dFp(X3p))

  F_pullback = F.get_tensor_pullback(p, Tp.type)
  out2 = F_pullback(Tp)(X1p, X2p, X3p)

  assert jnp.allclose(out1, out2)

  # Test something
  tensor1 = (f*T)(Fp)
  tensor2 = f(Fp)*T(Fp)
  assert jnp.allclose(tensor1.get_dense_coordinates(), tensor2.get_dense_coordinates())

  # Test properties from proposition 12.25
  # a)
  test1 = pullback_tensor_field(F, f*T) # Multiplied by an extra f(Fp)
  test2 = f(Fp)*pullback_tensor_field(F, T) # Correct

  tensor1 = test1(p)
  tensor2 = test2(p)

  assert jnp.allclose(tensor1.get_dense_coordinates(), tensor2.get_dense_coordinates())

  map1 = test1(X1, X2, X3)
  map2 = test2(X1, X2, X3)

  out1 = map1(p)
  out2 = map2(p)
  assert jnp.allclose(out1, out2)

  # b)
  test1 = tensor_field_product(pullback_tensor_field(F, T1), pullback_tensor_field(F, T2))
  test2 = pullback_tensor_field(F, tensor_field_product(T1, T2))

  out1 = test1(X1, X2, X3)(p)
  out2 = test2(X1, X2, X3)(p)
  assert jnp.allclose(out1, out2)

  # c)
  tensor_type = TensorType(0, 1)
  T1 = AutonomousTensorField(get_tensor_field_fun(N, tensor_type, k2), tensor_type, N)
  tensor_type = TensorType(0, 1)
  T2 = AutonomousTensorField(get_tensor_field_fun(N, tensor_type, k3), tensor_type, N)
  test1 = pullback_tensor_field(F, T1) + pullback_tensor_field(F, T2)
  test2 = pullback_tensor_field(F, T1 + T2)

  out1 = test1(X1)(p)
  out2 = test2(X1)(p)
  assert jnp.allclose(out1, out2)

  # # e) This is wrong in the book
  # def _G(p):
  #   assert p.shape == (5, 5)
  #   out = random.normal(k1, (5, 5))@p
  #   return out
  # G = Map(_G, domain=N, image=N)
  # GFp = G(F(p))

  # test1 = pullback_tensor_field(compose(F, G), T)
  # test2 = pullback_tensor_field(F, pullback_tensor_field(G, T))

  # out1 = test1(X1, X2, X3)(p)
  # out2 = test2(X1, X2, X3)(p)
  # assert jnp.allclose(out1, out2)


def run_all():
  jax.config.update("jax_enable_x64", True)

  # cotangent_tensor_product_test()
  tensor_field_tests()


if __name__ == "__main__":
  from debug import *
  run_all()