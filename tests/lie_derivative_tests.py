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
import nux
import src.util as util
from tests.vector_field_tests import get_vector_field_fun
from tests.tensor_tests import get_tensor_field_fun

################################################################################################################

def lie_derivative_tests():
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=4)
  p = random.normal(rng_key, (5,)); p = p/jnp.linalg.norm(p)

  # Construct a non-trivial tensor to test
  k1, k2, k3 = random.split(rng_key, 3)
  tensor_type = TensorType(0, 1)
  A1 = AutonomousTensorField(get_tensor_field_fun(M, tensor_type, k1), tensor_type, M)
  tensor_type = TensorType(0, 2)
  A2 = AutonomousTensorField(get_tensor_field_fun(M, tensor_type, k2), tensor_type, M)
  A = tensor_field_product(A1, A2)

  # Construct some vector fields to apply it to
  k1, k2, k3, k4 = random.split(k3, 4)
  X1 = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  X2 = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)
  X3 = AutonomousVectorField(get_vector_field_fun(M.dimension, k3), M)
  V = AutonomousVectorField(get_vector_field_fun(M.dimension, k4), M)
  test = A(X1, X2, X3)(p)

  # Create a function that we can test
  f = Map(lambda x: jnp.linalg.norm(jnp.sin(x)), domain=M, image=EuclideanManifold(dimension=1))

  # Proposition 12.32

  # b)
  test1 = lie_derivative(V, f*A)
  test2 = lie_derivative(V, f)*A + f*lie_derivative(V, A)

  out1 = test1(X1, X2, X3)(p)
  out2 = test2(X1, X2, X3)(p)
  assert jnp.allclose(out1, out2)

  # c)
  test1 = lie_derivative(V, A)
  test2 = tensor_field_product(lie_derivative(V, A1), A2) + tensor_field_product(A1, lie_derivative(V, A2))

  out1 = test1(X1, X2, X3)(p)
  out2 = test2(X1, X2, X3)(p)
  assert jnp.allclose(out1, out2)

  # d)
  test1 = lie_derivative(V, A(X1, X2, X3))
  test2 = lie_derivative(V, A)(X1, X2, X3)
  test2 += A(lie_derivative(V, X1), X2, X3)
  test2 += A(X1, lie_derivative(V, X2), X3)
  test2 += A(X1, X2, lie_derivative(V, X3))

  out1 = test1(p)
  out2 = test2(p)
  assert jnp.allclose(out1, out2)

  # Check that the exterior derivative passes through the lie derivative
  df = FunctionDifferential(f, M)
  test1 = lie_derivative(V, df)
  test2 = FunctionDifferential(lie_derivative(V, f), M)

  out1 = test1(X1)(p)
  out2 = test2(X1)(p)
  assert jnp.allclose(out1, out2)

  # Check composition of lie derivative
  W = X2
  test1 = lie_derivative(V, lie_derivative(W, A)) - lie_derivative(W, lie_derivative(V, A))
  test2 = lie_derivative(lie_bracket(V, W), A)

  out1 = test1(X1, X2, X3)(p)
  out2 = test2(X1, X2, X3)(p)
  assert jnp.allclose(out1, out2)


def run_all():
  jax.config.update("jax_enable_x64", True)
  lie_derivative_tests()

if __name__ == "__main__":
  from debug import *
  run_all()