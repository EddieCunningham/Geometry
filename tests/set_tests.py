from functools import partial
from typing import Callable
from typing import NewType
import src.util
import jax.numpy as jnp
from src.set import *
import src.util as util

def coordinate_test():
  """Check that we can instantiate a Coordinate object.
  """
  x = 0.5

def reals_test():
  """Check to see if membership works.
  """
  x = jnp.array([0.1, 0.4])
  R = Reals()

  assert x in R
  assert "a" not in R

def reals_dimension_test():
  """See if dimension checking works.
  """
  x = jnp.array([1.0, 2.0, 3.0])
  R3 = Reals(dimension=3)

  assert x in R3
  assert jnp.ones(3)*1.0 in R3
  assert jnp.ones((2, 3))*1.0 not in R3

def reals_subset_test():
  """Define a custom set membership function
  """

  class CustomSet(Reals):
    def __contains__(self, x):
      if super().__contains__(x):
        if jnp.sum(x**2) < 1.0:
          return True
      return False

  x = jnp.array([1.0, 2.0, 3.0])
  U = CustomSet(dimension=3)

  assert x not in U
  assert x*0 in U

################################################################################################################

def run_all():
  coordinate_test()
  reals_test()
  reals_dimension_test()
  reals_subset_test()

if __name__ == "__main__":
  from debug import *
  run_all()