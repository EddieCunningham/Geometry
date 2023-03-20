from functools import partial
from typing import Callable, List, Optional, Tuple
import src.util
from functools import partial
from copy import deepcopy
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
import src.util as util

__all__ = ["_LieGroupMixin",
           "LieGroup",
           "SemiDirectProductLieGroup",
           "semidirect_product",
           "internal_semidirect_product"]

class _LieGroupMixin(abc.ABC):
  """A smooth manifold that is also a group

  Attributes:
    atlas: Atlas providing coordinate representation.
    boundary: Optional boundary of the manifold
  """
  def get_identity_element(self) -> Point:
    """The identity element, e, of the Lie group

    Returns:
      e: The identity element
    """
    assert 0

  @property
  def e(self) -> Point:
    """The identity element, e, of the Lie group

    Returns:
      e: The identity element
    """
    return self.get_identity_element()

  def inverse(self, g: Point) -> Point:
    """The inverse map of the Lie group.  Satisfies m(g,I(g)) = e

    Args:
      g: An element of the Lie group

    Returns:
      g^{-1}
    """
    assert 0

  def multiplication_map(self, g: Point, h: Point) -> Point:
    """The multiplication map for the Lie group

    Args:
      g: An element of the Lie group
      h: An element of the Lie group

    Returns:
      m(g,h)
    """
    assert 0

  def left_translation_map(self, g: Point) -> Diffeomorphism[Point,Point]:
    """The left translation map L_g.  The map is L_g(h) = m(g,h)

    Args:
      g: An element of the Lie group

    Returns:
      L_g
    """
    def L_g(h: Point, inverse: bool=False)-> Point:
      if inverse == False:
        return self.multiplication_map(g, h)
      else:
        return self.multiplication_map(self.inverse(g), h)
    return Diffeomorphism(L_g, domain=self, image=self)

  def right_translation_map(self, g: Point) -> Diffeomorphism[Point,Point]:
    """The right translation map R_g.  The map is R_g(h) = m(h,g)

    Args:
      g: An element of the Lie group

    Returns:
      R_g
    """
    def R_g(h: Point, inverse: bool=False)-> Point:
      if inverse == False:
        return self.multiplication_map(h, g)
      else:
        return self.multiplication_map(h, self.inverse(g))
    return Diffeomorphism(R_g, domain=self, image=self)

  def left_action_map(self, g: Point, M: Manifold) -> Map[Point,Point]:
    """The left action map of G on M.  This is set to
       a translation map by default.

    Args:
      g: An element of the Lie group
      M: The manifold to apply an action on

    Returns:
      theta_g
    """
    def theta_g(p: Point) -> Point:
      return self.multiplication_map(g, p)
    return Map(theta_g, domain=M, image=M)

  def right_action_map(self, g: Point, M: Manifold) -> Map[Point,Point]:
    """The right action map of G on M.  This is set to
       a translation map by default.

    Args:
      g: An element of the Lie group
      M: The manifold to apply an action on

    Returns:
      theta_g
    """
    def theta_g(p: Point) -> Point:
      return self.multiplication_map(p, g)
    return Map(theta_g, domain=M, image=M)

  def conjugation_map(self, g: Point) -> Map[Point,Point]:
    """The conjugation map for g.  C_g: G -> G is given by C_g(h) = ghg^{-1}

    Args:
      g: An element of the Lie group

    Returns:
      C_g
    """
    def conjugate(h):
      g_inv = self.inverse(g)
      hg_inv = self.multiplication_map(h, g_inv)
      return self.multiplication_map(g, hg_inv)
    return Map(conjugate, domain=self, image=self)

class LieGroup(Manifold, _LieGroupMixin):
  Element = Point

################################################################################################################

class SemiDirectProductLieGroup(CartesianProductManifold, _LieGroupMixin):
  """The semi direct product of Lie groups N and H.  Elements are in the
     Cartesian product: (n,h) in N x H and the multiplication map is defined
     as (n,h)*(n',h') = (nL_h(n'), hh') where L_h: is a left action of H on N

  Attributes:
    N: Lie group 1
    H: Lie group 2
  """
  Element = Tuple[Point,Point]

  def __init__(self, N: LieGroup, H: LieGroup):
    """Create the object

    Args:
      N: Lie group 1
      H: Lie group 2
    """
    self.N = N
    self.H = H
    super().__init__(N, H)

  def __contains__(self, p: Element) -> bool:
    """Check if p is a part this object

    Args:
      p: Point

    Returns:
      Whether or not p in in the semi-direct product
    """
    n, h = p
    return (n in self.N) and (h in self.H)

  def get_identity_element(self) -> Point:
    """The identity element, e, of the Lie group

    Returns:
      e: The identity element
    """
    return (self.N.get_identity_element(), self.H.get_identity_element())

  def inverse(self, g: Point) -> Point:
    """The inverse map of the Lie group.  Satisfies m(g,I(g)) = e

    Args:
      g: An element of the Lie group

    Returns:
      g^{-1}
    """
    # Unpack the point
    n, h = g

    n_inv = self.N.inverse(n)
    h_inv = self.H.inverse(h)

    # Construct the left action map
    theta_h_inv = self.H.left_action_map(h_inv, self.N)
    return (theta_h_inv(n_inv), h_inv)

  def multiplication_map(self, g1: Point, g2: Point) -> Point:
    """The multiplication map for the Lie group

    Args:
      g: An element of the Lie group
      h: An element of the Lie group

    Returns:
      m(g,h)
    """
    # Unpack the point
    n1, h1 = g1
    n2, h2 = g2

    # Construct the left action map
    theta_h1 = self.H.left_action_map(h1, self.N)

    # Construct the elements of the tuple
    n = self.N.multiplication_map(n1, theta_h1(n2))
    h = self.H.multiplication_map(h1, h2)
    return (n, h)

def semidirect_product(N: LieGroup, H: LieGroup) -> LieGroup:
  """The semidirect product of H and N.  Elements are in the Cartesian product:
     (n,h) in N x H and the multiplication map is defined as
     (n,h)*(n',h') = (nL_h(n'), hh') where L_h: is a left action of H on N

  Args:
    N: A Lie group
    H: A Lie group

  Returns:
    NH: The semi-direct product of N and H
  """
  return SemiDirectProductLieGroup(N, H)

def internal_semidirect_product(N: LieGroup, H: LieGroup) -> LieGroup:
  """Returns the internal semi-direct product of 2 Lie groups.  Has the
  property that the map (n,h) -> nh is a Lie group isomorphism between
  the semidirect product of N and H and NH.

  Args:
    N: A Lie group
    H: A Lie group

  Returns:
    NH: The semi-direct product of N and H
  """
  H_copy = deepcopy(H)

  # Change the left action map of H to the conjugate map
  def left_action_map(self, g: Point, M: Manifold):
    def conjugate(h):
      g_inv = self.inverse(g)
      hg_inv = self.multiplication_map(h, g_inv)
      return self.multiplication_map(g, hg_inv)
    return Map(conjugate, domain=M, image=M)

  H_copy.left_action_map = left_action_map

  return semidirect_product(N, H_copy)

################################################################################################################
