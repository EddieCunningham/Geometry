from functools import partial
from typing import Callable
from typing import NewType
import src.util
import jax.numpy as jnp
from src.set import *
from src.vector import EuclideanSpace
import src.util as util

def coordinate_test():
  """Check that we can instantiate a Coordinate object.
  """
  x = 0.5

def reals_test():
  """Check to see if membership works.
  """
  x = jnp.array([0.1, 0.4])
  R = EuclideanSpace(dimension=2)

  assert x in R
  assert "a" not in R

def reals_dimension_test():
  """See if dimension checking works.
  """
  x = jnp.array([1.0, 2.0, 3.0])
  R3 = EuclideanSpace(dimension=3)

  assert x in R3
  assert jnp.ones(3)*1.0 in R3
  assert jnp.ones((2, 3))*1.0 not in R3

def reals_subset_test():
  """Define a custom set membership function
  """

  class CustomSet(EuclideanSpace):
    def contains(self, x):
      if super().contains(x):
        if jnp.sum(x**2) < 1.0:
          return True
      return False

  x = jnp.array([1.0, 2.0, 3.0])
  U = CustomSet(dimension=3)

  assert x not in U
  assert x*0 in U

def set_intersect_test():

  class CustomSet1(EuclideanSpace):
    def contains(self, x):
      if super().contains(x):
        if jnp.sqrt(jnp.sum(x**2)) < 1.0:
          return True
      return False

  class CustomSet2(EuclideanSpace):
    def contains(self, x):
      if super().contains(x):
        if jnp.any(x < 0):
          return False
        else:
          return True
      return False

  x = jnp.array([0.1, 0.6, 0.2])
  y = jnp.array([0.1, -0.6, 0.2])
  z = jnp.array([0.1, 10.6, 0.2])

  A = CustomSet1(dimension=3)
  B = CustomSet2(dimension=3)
  C = A.intersect_with(B)

  assert x in A
  assert x in B
  assert x in C

  assert y in A
  assert y not in B
  assert y not in C

  assert z not in A
  assert z in B
  assert z not in C

################################################################################################################

def run_all():
  coordinate_test()
  reals_test()
  reals_dimension_test()
  reals_subset_test()
  set_intersect_test()

if __name__ == "__main__":
  from debug import *
  run_all()
