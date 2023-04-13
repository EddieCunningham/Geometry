from functools import partial
from typing import Callable, TypeVar, Tuple, Generic, List
import src.util
import jax.numpy as jnp
import abc
import src.util as util
import copy

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

  def __contains__(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    return self.contains(p)

  @abc.abstractmethod
  def contains(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    return True

  def intersect_with(self, other: "Set") -> "Set":
    """Intersect this set with another set.

    Args:
      other: Another set.

    Returns:
      A shallow copy of this set with a new contains function.
    """
    instance = self
    def new_contains(self, p: Point) -> bool:
      in_self = instance.contains(p)
      in_other = other.contains(p)
      return in_self and in_other

    # Create a shallow copy of this set and replace the contains method
    new_set = copy.copy(self)
    new_set.contains = new_contains.__get__(new_set)
    return new_set

class EmptySet(Set):
  """The empty set.  Nothing is included in this

  Attributes:
    membership_function: Function that determines membership.
  """

  def contains(self, p: Point) -> bool:
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

    from src.manifold import Chart, Atlas
    self.chart = Chart(lambda x, inverse=False: x, domain=self, image=self)
    self.atlas = Atlas([self.chart])
    super().__init__()

  def __str__(self) -> str:
    """The string representation of this manifold

    Returns:
      A string
    """
    return f"{type(self)}(dimension={self.dimension})"

  def contains(self, p: Coordinate) -> bool:
    """Checks to see if p exists in this set and is real.

    Args:
      p: Test point.

    Returns:
      True if p is in the set and real, False otherwise.
    """
    if isinstance(p, float) and ((self.dimension is None) or (self.dimension <= 1)):
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

    return super().contains(p)

  def __eq__(self, other: Set) -> bool:
    """Checks to see if another set is the set of reals.
    Because we construct these often, it doesn't make
    sense to compare the instances to each other.

    Args:
      other: A set

    Returns:
      boolean
    """
    from src.instances.manifolds import EuclideanManifold

    if isinstance(other, EuclideanManifold):
      if self.dimension is None:
        return other.dimension <= 1
      return self.dimension == other.dimension

    type_check = isinstance(other, Reals)

    if other.dimension is None:
      if self.dimension:
        dim_check = self.dimension <= 1
      else:
        dim_check = True # Both dimensions are None
    else:
      dim_check = self.dimension == other.dimension
    return dim_check and type_check

  def get_chart_for_point(self, p: Point) -> "Chart":
    """Get a chart to use at point p

    Args:
      The input point

    Returns:
      The chart that contains p in its domain
    """
    return self.chart

################################################################################################################

class CartesianProductSet(Set):
  """Set that is the cartesian product of multiple sets

  Attributes:
    membership_function: Function that determines membership.
  """
  Element = List[Point]

  def __init__(self, *sets: List[Set]):
    """Create the cartesian product of the sets

    Args:
      sets: A list of set objects
    """
    self.sets = sets

  def contains(self, p: Element) -> bool:
    """Check if p is a cartesian product

    Args:
      p: The input point

    Returns:
      Whether or not p is in the cartesian product
    """
    assert len(p) == len(self.sets)
    return all([_p in _set for _p, _set in zip(p, self.sets)])

def set_cartesian_product(*sets: List[Set]) -> CartesianProductSet:
  """Make a set that has elements (a, b, c, ...., K)

  Args:
    sets: A list of sets

  Returns:
    Cartesian product of the sets
  """
  return CartesianProductSet(*sets)
