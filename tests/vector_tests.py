from functools import partial
from typing import Callable
from typing import NewType
import src.util
import jax.numpy as jnp
import jax.random as random
from src.set import *
from src.vector import *
import src.util as util
import jax

def euclidean_space_tests():
  # Test equality
  R3 = EuclideanSpace(dimension=3)
  R3_2 = EuclideanSpace(dimension=3)
  R4 = EuclideanSpace(dimension=4)
  assert R3 == R3_2
  assert R3 != R4

################################################################################################################

def test_vector_creation():
  R4 = VectorSpace(dimension=4)

  rng_key = random.PRNGKey(0)
  k1, k2, k3 = random.split(rng_key, 3)

  # Create a vector
  x = random.normal(k1, (R4.dimension,))
  v = Vector(x, R4)

  # Check that it is in R4
  assert v in R4

  # Create another vector and check that it isn't in R4
  x = random.normal(k2, (R4.dimension + 1,))
  v2 = Vector(x, R4)
  assert not (v2 in R4)

def test_vector_operations():
  R4 = VectorSpace(dimension=4)

  rng_key = random.PRNGKey(0)
  k1, k2, k3 = random.split(rng_key, 3)

  # Check that we can perform operations on vectors
  coords = random.normal(k3, (3, R4.dimension))
  a = Vector(coords[0], R4)
  b = Vector(coords[1], R4)
  c = Vector(coords[2], R4)

  # Check that we can add them together
  d = a + b
  assert d in R4

  # Try multiplying by scalars
  s, t = 2.1, 1.3
  e = s*a + t*b - c
  assert e in R4

def test_vmap():

  # Try vmapping these operations
  
  def create_vector(rng_key):
    R4 = VectorSpace(dimension=4)
    coords = random.normal(rng_key, (2, R4.dimension,))
    a = Vector(coords[0], R4)
    b = Vector(coords[1], R4)
    return (a + b).x
  
  rng_key = random.PRNGKey(0)
  keys = random.split(rng_key, 3)
  out = jax.vmap(create_vector)(keys)

################################################################################################################

def run_all():
  euclidean_space_tests()
  test_vector_creation()
  test_vector_operations()
  test_vmap()

if __name__ == "__main__":
  from debug import *
  run_all()