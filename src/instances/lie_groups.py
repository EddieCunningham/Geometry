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
           "GLLieG",
           "GLRn",
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

  def contains(self, p: Point) -> bool:
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
  """GL(V)

  Attributes:
    dim: Dimensionality
  """
  Element = LinearMap

  def __init__(self, V: VectorSpace):
    """GL(V) is the set of all invertible linear maps
    from V to V.

    Args:
      V: A vector space
    """
    self.vector_space = V
    dim = self.vector_space.dimension
    super().__init__(dimension=dim**2)

  def contains(self, g: LinearMap) -> bool:
    """Check if g is an invertible matrix

    Args:
      g: Test point.

    Returns:
      True if g is in the set, False otherwise.
    """
    type_check = isinstance(g, LinearMap)
    domain_check = g.domain == self.vector_space
    image_check = g.image == self.vector_space
    return domain_check and image_check

  def inverse(self, g: LinearMap) -> LinearMap:
    """The inverse map of the Lie group, which is the matrix inverse

    Args:
      g: An element of the Lie group

    Returns:
      g^{-1}
    """
    return g.get_inverse()

  def multiplication_map(self, g: Point, h: Point) -> Point:
    """The multiplication map for the Lie group, matrix multiplication

    Args:
      g: An element of the Lie group
      h: An element of the Lie group

    Returns:
      m(g,h)
    """
    return compose(g, h)

################################################################################################################

class GLLieG(GeneralLinearGroup):
  """GL(Lie(G))

  Attributes:
    G: A Lie group
  """
  Element = LinearMap["LeftInvariantVectorField","LeftInvariantVectorField"]

  def __init__(self, G: LieGroup):
    """Create the space of invertible nxn matrices

    Args:
      dim: Dimension
    """
    self.G = G
    super().__init__(G)

  def contains(self, p: LinearMap) -> bool:
    """Check if p is an invertible matrix

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    from src.lie_algebra import LieAlgebra
    type_check = isinstance(p, LinearMap)
    domain_check = isinstance(p.domain, LieAlgebra)
    image_check = isinstance(p.image, LieAlgebra)
    return type_check and domain_check and image_check

  def get_atlas(self) -> Atlas:
    """The chart outputs the flattened matrix

    Returns:
      atlas: The atlas object
    """
    from src.lie_algebra import LieAlgebra, LeftInvariantVectorField
    def phi(x, inverse=False):
      if inverse == False:
        assert isinstance(x, LinearMap)
        assert isinstance(x.domain, LieAlgebra)
        assert isinstance(x.image, LieAlgebra)

        # Get a basis for the domain's tangent space at e
        basis = x.domain.TeG.get_basis()

        Js = []
        for Xe in basis:
          # Get the corresponding left invariant vector fields
          X = x.domain.get_left_invariant_vector_field(Xe)

          # Get the coordinates of the map at X
          J = x.get_differential(X).get_coordinates()
          Js.append(J)

        test = x(X)(self.G.e)
        import pdb; pdb.set_trace()
        # TODO: FIGURE THIS OUT

        return x.ravel()
      return x.reshape((self.N, self.N))

    return Atlas([Chart(phi=phi, domain=self, image=Reals(dimension=self.dimension))])

  def get_identity_element(self) -> Point:
    """The identity matrix

    Returns:
      e: The identity element
    """
    return Map(lambda X: X, domain=self.lieG, image=self.lieG)

  def inverse(self, g: Point) -> Point:
    """The inverse map of the Lie group, which is the matrix inverse

    Args:
      g: An element of the Lie group

    Returns:
      g^{-1}
    """
    assert 0

  def multiplication_map(self, g: Point, h: Point) -> Point:
    """The multiplication map for the Lie group, matrix multiplication

    Args:
      g: An element of the Lie group
      h: An element of the Lie group

    Returns:
      m(g,h)
    """
    return compose(g, h)

  def get_lie_algebra(self) -> "LieAlgebra":
    """Get the Lie algebra associated with this Lie group.  This is the
    tangent space at the identity and it is equipped with a Lie bracket.

    Returns:
      Lie algebra of this Lie group
    """
    assert 0

################################################################################################################

class GLRn(GeneralLinearGroup):
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
    V = EuclideanManifold(self.N)
    super().__init__(V)

  def contains(self, p: Point) -> bool:
    """Check if p is an invertible matrix

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    shape_condition = p.shape == (self.N, self.N)
    det_condition = True
    if util.GLOBAL_CHECK:
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
    return SpaceOfMatrices(self)

class GLp(GLRn):
  """GL+(n,R) is the Lie group of matrices with positive determinant

  Attributes:
    dim: Dimension
  """
  Element = Matrix

  def contains(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    if super().contains(p) == False:
      return False

    if util.GLOBAL_CHECK == False:
      return True
    return jnp.linalg.slogdet(p)[0] > 0

class OrthogonalGroup(GLRn):
  """O(n,R) is the Lie group of orthogonal matrices

  Attributes:
    dim: Dimension
  """
  Element = Matrix

  def contains(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    if super().contains(p) == False:
      return False

    if util.GLOBAL_CHECK == False:
      return True
    return jnp.allclose(jnp.linalg.svd(p, compute_uv=False), 1.0)

  def get_lie_algebra(self) -> "LieAlgebra":
    """Get the Lie algebra associated with this Lie group.  This is the
    tangent space at the identity and it is equipped with a Lie bracket.

    Returns:
      Lie algebra of this Lie group
    """
    from src.lie_algebra import SpaceOfMatrices
    class SpaceOfSkewSymmetricMatrices(SpaceOfMatrices):
      def contains(self, p: Point) -> bool:
        if super().contains(p) == False:
          return False
        if util.GLOBAL_CHECK == False:
          return True
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
