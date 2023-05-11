from functools import partial
from typing import Callable, TypeVar, Tuple, Generic, List
import src.util
import jax.numpy as jnp
import abc
import src.util as util
import copy

__all__ = ["Point",
           "Set",
           "EmptySet",
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

class CartesianProductSet(Set):
  """Set that is the cartesian product of multiple sets

  Attributes:
    membership_function: Function that determines membership.
  """
  Element = List[Point]
  sets: List[Set]

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
    for p, s in zip(p, self.sets):
      if p not in s:
        return False
    return True

def set_cartesian_product(*sets: List[Set]) -> CartesianProductSet:
  """Make a set that has elements (a, b, c, ...., K)

  Args:
    sets: A list of sets

  Returns:
    Cartesian product of the sets
  """
  return CartesianProductSet(*sets)
