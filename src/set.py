from functools import partial
from typing import Callable, TypeVar, Tuple, Generic
import src.util
import jax.numpy as jnp
import abc
import src.util as util

__all__ = ["Point",
           "Coordinate",
           "Matrix",
           "InvertibleMatrix",
           "Set",
           "EmptySet",
           "Reals",
           "CartesianProductSet",
           "set_cartesian_product"]

"""Abstract point.  Will represent a point on a manifold."""
Point = TypeVar("Point")

class Set(abc.ABC):
  """Set object.  Can contain things.

  Attributes:
    membership_function: Function that determines membership.
  """
  Element = Point

  def __init__(self):
    """Creates a new Set object
    """
    pass

  @abc.abstractmethod
  def __contains__(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    return True

class EmptySet(Set):
  """The empty set.  Nothing is included in this

  Attributes:
    membership_function: Function that determines membership.
  """

  def __contains__(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      Always False.
    """
    return False

################################################################################################################

"""Coordinate.  Will represent a point on Euclidean space and can be
  implemented using numpy arrays.  Important to make this distinction
  because we will always want to do computations in Euclidean space.
"""
Coordinate = TypeVar("Coordinate", bound=jnp.array)

class Matrix(Generic[Coordinate]):
  pass

class InvertibleMatrix(Generic[Coordinate]):
  pass

class Reals(Set):
  """Set of real numbers.  We can do computations here.

  Attributes:
    dimension: dimension of this space.
    membership_function: Function that determines membership.
  """
  Element = Coordinate

  def __init__(self, dimension: int=None):
    """Creates a new Euclidean space.  Can optionally set the dimension as well.

    Args:
      dimension: An integer representing the dimensionality of the space.
      membership_function: Function that determines membership.
    """
    self.dimension = dimension
    super().__init__()

  def __contains__(self, p: Coordinate) -> bool:
    """Checks to see if p exists in this set and is real.

    Args:
      p: Test point.

    Returns:
      True if p is in the set and real, False otherwise.
    """
    if isinstance(p, float) and self.dimension <= 1:
      return True

    # Must be a coordinate type
    if isinstance(p, jnp.ndarray) == False:
      return False

    # Must be a vector or a scalar
    if p.ndim > 1:
      return False

    # Optionally, must have a fixed dimensionality
    if self.dimension is not None:
      if self.dimension > 1:
        if p.shape[-1] != self.dimension:
          return False
      else:
        if len(p.shape) == 0:
          pass # This is ok
        elif p.shape[-1] != self.dimension:
          return False

    return super().__contains__(p)

################################################################################################################

class CartesianProductSet(Set):
  """Set that is the cartesian product of 2 sets

  Attributes:
    membership_function: Function that determines membership.
  """
  Element = Tuple[Point,Point]

  def __init__(self, a: Set, b: Set):
    """Create the cartesian product of a and b

    Args:
      a: Set a
      b: Set b
    """
    self.set_a = a
    self.set_b = b

  def __contains__(self, p: Element) -> bool:
    """Check if p is a cartesian product

    Args:
      p: The input point

    Returns:
      Whether or not p is in the cartesian product
    """
    pa, pb = p
    return (pa in self.set_a) and (pb in self.set_b)

def set_cartesian_product(a: Set, b: Set) -> CartesianProductSet:
  """Make a set that has elements (a,b)

  Args:
    a: Set A
    b: Set B

  Returns:
    Cartesian product AxB
  """
  return CartesianProductSet(a, b)