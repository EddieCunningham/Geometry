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

################################################################################################################

def cotangent_tensor_product_test():
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=4)
  p = random.normal(rng_key, (5,)); p = p/jnp.linalg.norm(p)

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

  import pdb; pdb.set_trace()


def run_all():
  jax.config.update("jax_enable_x64", True)

  # cotangent_tensor_product_test()
  tensor_field_tests()

  assert 0


  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=5)
  p = random.normal(rng_key, (5,)); p = p/jnp.linalg.norm(p)



  # Construct 4 cotangent vectors
  cotangent_coords = random.normal(rng_key, (3, M.dim))
  coTpM = CotangentSpace(p, M)
  tangent_vectors = [CotangentVector(coords, coTpM) for coords in cotangent_coords]

  # Construct 3 tangent vectors
  tangent_coords = random.normal(rng_key, (2, M.dim))
  TpM = TangentSpace(p, M)
  tangent_vectors = [TangentVector(coords, TpM) for coords in tangent_coords]

  # Create a tensor product


  # Create a tensor that accepts these that is equal to the tensor product


if __name__ == "__main__":
  from debug import *
  run_all()