from functools import partial
from typing import Callable, List, Optional, Tuple, Generic
import src.util
from functools import partial
from copy import deepcopy
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
from src.lie_group import *
from src.instances.manifolds import *
from src.vector import *
import src.util as util

__all__ = ["RealLieGroup",
           "GeneralLinearGroup",
           "GLp",
           "OrthogonalGroup",
           "EuclideanGroup"]

class RealLieGroup(LieGroup):
  """Lie group under addition

  Attributes:
    dim: Dimensionality
  """
  Element = Coordinate

  def __init__(self, dimension: int):
    """Create Euclidean space

    Args:
      dimension: Dimension
    """
    self.space = Reals(dimension=dimension)
    super().__init__(dimension=dimension)

  def __contains__(self, p: Point) -> bool:
    return p in self.space

  def get_atlas(self) -> Atlas:
    """Return the chart for the Lie group

    Returns:
      atlas: The atlas object
    """
    return Atlas([Chart(phi=lambda x, inverse=False: x, domain=self.space, image=self.space)])

  def get_identity_element(self) -> Point:
    """The identity element

    Returns:
      e: The identity element
    """
    return jnp.zeros(self.dimension)

  def inverse(self, g: Point) -> Point:
    """The inverse map of the Lie group

    Args:
      g: An element of the Lie group

    Returns:
      g^{-1}
    """
    return -g

  def multiplication_map(self, g: Point, h: Point) -> Point:
    """The multiplication map for the Lie group

    Args:
      g: An element of the Lie group
      h: An element of the Lie group

    Returns:
      m(g,h)
    """
    return g + h

################################################################################################################

class GeneralLinearGroup(LieGroup):
  """GL(n,R)

  Attributes:
    dim: Dimensionality
  """
  Element = InvertibleMatrix

  def __init__(self, dim: int):
    """Create the space of invertible nxn matrices

    Args:
      dim: Dimension
    """
    self.N = dim
    super().__init__(dimension=dim**2)

  def __contains__(self, p: Point) -> bool:
    """Check if p is an invertible matrix

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    shape_condition = p.shape == (self.N, self.N)
    det_condition = jnp.linalg.slogdet(p)[1] > -1e5
    return shape_condition and det_condition

  def get_atlas(self) -> Atlas:
    """The chart outputs the flattened matrix

    Returns:
      atlas: The atlas object
    """
    def phi(x, inverse=False):
      if inverse == False:
        return x.ravel()
      return x.reshape((self.N, self.N))

    return Atlas([Chart(phi=phi, domain=self, image=Reals(dimension=self.dimension))])

  def get_identity_element(self) -> Point:
    """The identity matrix

    Returns:
      e: The identity element
    """
    return jnp.eye(self.N)

  def inverse(self, g: Point) -> Point:
    """The inverse map of the Lie group, which is the matrix inverse

    Args:
      g: An element of the Lie group

    Returns:
      g^{-1}
    """
    return jnp.linalg.inv(g)

  def multiplication_map(self, g: Point, h: Point) -> Point:
    """The multiplication map for the Lie group, matrix multiplication

    Args:
      g: An element of the Lie group
      h: An element of the Lie group

    Returns:
      m(g,h)
    """
    return g@h

  def get_lie_algebra(self) -> "LieAlgebra":
    """Get the Lie algebra associated with this Lie group.  This is the
    tangent space at the identity and it is equipped with a Lie bracket.

    Returns:
      Lie algebra of this Lie group
    """
    from src.lie_algebra import SpaceOfMatrices
    return SpaceOfMatrices(dim=self.N)

################################################################################################################

class GLp(GeneralLinearGroup):
  """GL+(n,R) is the Lie group of matrices with positive determinant

  Attributes:
    dim: Dimension
  """
  Element = Matrix

  def __contains__(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    if super().__contains__(p) == False:
      return False
    return jnp.linalg.slogdet(p)[0] > 0

class OrthogonalGroup(GeneralLinearGroup):
  """O(n,R) is the Lie group of orthogonal matrices

  Attributes:
    dim: Dimension
  """
  Element = Matrix

  def __contains__(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    if super().__contains__(p) == False:
      return False

    return jnp.allclose(jnp.linalg.svd(p, compute_uv=False), 1.0)

  def get_lie_algebra(self) -> "LieAlgebra":
    """Get the Lie algebra associated with this Lie group.  This is the
    tangent space at the identity and it is equipped with a Lie bracket.

    Returns:
      Lie algebra of this Lie group
    """
    from src.lie_algebra import SpaceOfMatrices
    class SpaceOfSkewSymmetricMatrices(SpaceOfMatrices):
      def __contains__(self, p: Point) -> bool:
        if super().__contains__(p) == False:
          return False
        return jnp.allclose(p, -p.T)
    return SpaceOfSkewSymmetricMatrices(dim=self.N)

################################################################################################################

class EuclideanGroup(SemiDirectProductLieGroup):
  """"Make the Euclidean group.  This is the semi direct product of
  the set of reals and the orthogonal group of the same dimension.

  Attributes:
  """
  Element = Tuple[Vector,Matrix]

  def __init__(self, dim: int):
    """Create the euclidean group of dimension dim.

    Args:
      dim: dimension
    """
    self.dim = dim
    self.R = RealLieGroup(dim)
    self.On = OrthogonalGroup(dim)
    super().__init__(self.R, self.On)

  def left_action_map(self, g: Point, M: Manifold) -> Map[Point,Point]:
    """The left action map of G on M.  This is set to
       a translation map by default.

    Args:
      g: An element of the Lie group
      M: The manifold to apply an action on

    Returns:
      theta_g
    """
    assert isinstance(M, RealLieGroup)
    b, A = g
    def theta_g(x):
      return b + A@x
    return Map(theta_g, domain=M, image=M)

################################################################################################################
