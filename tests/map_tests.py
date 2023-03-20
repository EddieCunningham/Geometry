from functools import partial
from typing import Callable
from typing import NewType
import src.util
import jax.numpy as jnp
from src.set import *
from src.map import *
from src.instances.manifolds import *
import src.util as util

def function_test():
  """See if we can initialize a map
  """
  def f(x):
    return jnp.concatenate([x, 2*x], axis=-1)

  domain = Reals(dimension=2)
  image = Reals(dimension=4)
  function = Function(f, domain=domain, image=image)

  x = jnp.array([0.0, 1.0])
  y = function(x)
  assert jnp.allclose(y, jnp.array([0.0, 1.0, 0.0, 2.0]))

def function_domain_image_test():
  """Try to pass wrong inputs to the function
  """
  def f(x):
    return jnp.concatenate([x, 2*x], axis=-1)

  domain = Reals(dimension=2)
  image = Reals(dimension=4)
  function = Function(f, domain=domain, image=image)

  x = jnp.array([0.0, 1.0])
  out = function(x)

  try:
    function(f(x))
    assert 0
  except:
    pass

def invertible_function_test():
  """Make sure that the inverse functions work
  """

  def f(x, inverse=False):
    if inverse:
      return x - 1.0
    return x + 1.0

  x = jnp.array([0.0, 1.0])

  domain = Reals()
  image = Reals()
  function = InvertibleFunction(f, domain=domain, image=image)
  inverse_function = function.get_inverse()

  y = function(x)
  x_reconstr1 = function.inverse(y)
  x_reconstr2 = inverse_function(y)
  assert jnp.allclose(x_reconstr1, x_reconstr2)

def composition_test():
  """Check that we can compose maps correctly
  """

  class A(Set):
    def __contains__(self, p):
      return isinstance(p, str)

  class B(Set):
    def __contains__(self, p):
      return isinstance(p, int)

  class C(Set):
    def __contains__(self, p):
      return isinstance(p, float)

  class D(Set):
    def __contains__(self, p):
      return isinstance(p, bool)

  def a_to_b(p):
    return 1

  def b_to_c(p):
    return 2.4

  def c_to_d(p):
    return True

  map1 = Map(a_to_b, domain=A(), image=B())
  map2 = Map(b_to_c, domain=B(), image=C())
  map3 = Map(c_to_d, domain=C(), image=D())

  out1 = map1("a")
  out2 = map2(out1)
  out3 = map3(out2)

  map123 = compose(map3, map2, map1)
  out123 = map123("a")

def invertible_composition_test():
  """Check that we can compose maps correctly
  """

  class A(Set):
    def __contains__(self, p):
      return isinstance(p, str)

  class B(Set):
    def __contains__(self, p):
      return isinstance(p, int)

  class C(Set):
    def __contains__(self, p):
      return isinstance(p, float)

  class D(Set):
    def __contains__(self, p):
      return isinstance(p, bool)

  def a_to_b(p, inverse=False):
    return 1 if inverse == False else "a"

  def b_to_c(p, inverse=False):
    return 2.4 if inverse == False else 1

  def c_to_d(p, inverse=False):
    return True if inverse == False else 2.4

  map1 = Diffeomorphism(a_to_b, domain=A(), image=B())
  map2 = Diffeomorphism(b_to_c, domain=B(), image=C())
  map3 = Diffeomorphism(c_to_d, domain=C(), image=D())

  out1 = map1("a")
  out2 = map2(out1)
  out3 = map3(out2)

  map123 = compose(map3, map2, map1)
  out123 = map123("a")
  out123_inv = map123.inverse(out123)

################################################################################################################

def run_all():
  function_test()
  function_domain_image_test()
  invertible_function_test()
  composition_test()
  invertible_composition_test()

if __name__ == "__main__":
  from debug import *
  run_all()