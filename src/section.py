from functools import partial
from typing import Callable, List, Optional, Union
import src.util
from functools import partial
from copy import deepcopy
import jax
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
import src.util as util

__all__ = ["Section",
           "BundleHomomorphismSection",
           "apply_bundle_homomorphism_to_section"]

class Section(Map[Input,Output], abc.ABC):
  """If pi: M -> N is a continuous map, then a section of pi is a continuous right inverse
     for pi, so sigma: N -> M s.t. pi(sigma(.)) = Id(.)

  Attributes:
    pi: The left inverse of sigma.
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  def __init__(self, pi: Map[Output,Input]):
    """Creates a new section.

    Args:
      pi: Map that this is a seciton of
    """
    self.pi = pi
    self.manifold = self.pi.image
    self.total_space = self.pi.domain
    super().__init__(self.__call__, domain=self.pi.image, image=self.pi.domain)

  @abc.abstractmethod
  def apply_to_point(self, p: Input) -> Output:
    """Evaluate the section at a point.  Output should be a
    point on the total space.

    Args:
      p: Point on the manifold.

    Returns:
      Point on total space.
    """
    pass

  def __call__(self, p: Point) -> Output:
    """Evaluate the section at a point.  Output should be a
    point on the total space.  This function is not abstract
    because classes that inherit from this can have different
    functionality.

    Args:
      p: Point on the manifold.

    Returns:
      Point on total space.
    """
    return self.apply_to_point(p)

  def __add__(self, Y: "Section") -> "Section":
    """Add two sections together

    Args:
      Y: Another section

    Returns:
      X + Y
    """
    # Ensure that X and Y are compatable
    assert self.manifold == Y.manifold
    # assert self.total_space == Y.total_space

    class SectionSum(type(self)):
      def __init__(self, X, Y, pi):
        self.X = X
        self.Y = Y
        Section.__init__(self, pi)

      def apply_to_point(self, p: Input) -> Output:
        return self.X(p) + self.Y(p)

    return SectionSum(self, Y, self.pi)

  def __radd__(self, Y: "Section") -> "Section":
    return self + Y

  def __rmul__(self, f: Union[Map,float]) -> "Section":
    """Multiply a section with a scalar or function. fX

    Args:
      f: Another map or a scalar

    Returns:
      fX
    """
    is_map = isinstance(f, Map)
    is_scalar = f in Reals(dimension=1)

    assert is_map or is_scalar

    class SectionRHSProduct(type(self)):
      def __init__(self, X, pi):
        self.X = X
        self.lhs = f
        self.is_float = f in Reals(dimension=1)
        Section.__init__(self, pi)

      def apply_to_point(self, p: Input) -> Output:
        fp = self.lhs if self.is_float else self.lhs(p)
        Xp = self.X(p)
        return fp*Xp

    return SectionRHSProduct(self, self.pi)

  def __neg__(self):
    """Negate the vector field

    Returns:
      -X
    """
    return -1.0*self

  def __sub__(self, Y: "Section") -> "Section":
    return self + -Y

################################################################################################################

class BundleHomomorphismSection(Section):
  """A section induced by a bundle homomorphism.
  Is linear over real functions.

  Attributes:
    F: The bundle homorphism
    s: The section on E
  """
  def __init__(self, F: Map, s: Section):
    """Create a space of sections

    Args:
      pi: The map that this is a secion of
    """
    self.s = s
    self.F = F
    pi = self.F.image.get_projection_map()
    super().__init__(pi)

  def apply_to_point(self, p: Input) -> Output:
    """Evaluate the section at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Section at p.
    """
    return self.F(self.s(p))

def apply_bundle_homomorphism_to_section(F: Map, s: Section):
    """Apply a bundle homorphism to a section.  F must be a bundle homomorphism,
    meaning that it is linear in the fibers and correspond to some function over
    the manifold.

    Args:
      F: A bundle homomorphism.
      s: A section of the bundle.
    """
    return BundleHomomorphismSection(F, s)
