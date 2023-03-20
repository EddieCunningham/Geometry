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
from src.tangent import *
from src.cotangent import *
from src.vector_field import *
from src.instances.manifolds import Vector, VectorSpace, EuclideanManifold
import diffrax
import abc

__all__ = ["FiberBundle",
           "VectorBundle",
           "ProductBundle",
           "TangentBundle",
           "CotangentBundle",
           "FrameBundle"]

################################################################################################################

class Fiber(Generic[Point]):
  pass

class FiberBundle(abc.ABC):
  """A fiber bundle is a space that is locally a product space.  Each point, p, of the manifold
  has an associated fiber, F_p.  The projection map, pi, goes from the total space to the base space.
  All computations will be done using local trivializations.

  Attributes:
    F: The fiber F.
    M: The base space.
    E: The total space which looks looks like MxF around points.
  """
  Element = Tuple[Point,Fiber]

  def __init__(self, M: Manifold, F: Fiber):
    """Create a fiber bundle

    Attributes:
      F: The fiber F.
      M: The base space.
    """
    self.fiber = F
    self.manifold = M

    # DIMENSION OF TOTAL SPACE, NOT THE BUNDLE!
    self.dimension = F.dimension + M.dimension

  @abc.abstractmethod
  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space.

    Returns:
      The map x -> p, x in E, p in M
    """
    pass

  @abc.abstractmethod
  def local_trivialization(self, p: Point) -> "ProductBundle":
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      (p, f), where p is in the base manifold and f is in the fiber
    """
    pass

  def __contains__(self, x: Element) -> bool:
    """Checks to see if x exists in the manifold.  This is the case if
       the point is in the domain of any chart

    Args:
      x: Test point.

    Returns:
      True if p is in the manifold, False otherwise.
    """
    p = self.get_projection_map()(x)
    UxF = self.local_trivialization(p)
    return x in UxF

  def get_chart_for_point(self, x: Element) -> Chart:
    """Get a chart to use at point x in the total space.  Do this by
    getting a chart of the local trivialization at pi(x)

    Args:
      The input point

    Returns:
      The chart that contains x in its domain
    """
    p = self.get_projection_map()(x)
    UxF = self.local_trivialization(p)
    return UxF.get_chart_for_point(x)

################################################################################################################

class ProductBundle(CartesianProductManifold, FiberBundle):
  """Trivial bundle is just the product of the base space and fiber.

  Attributes:
    MxF: The cartesian product of M and F
    See ProductManifold for more details
  """
  Element = Tuple[Point,Fiber]

  def __init__(self, M: Manifold, F: Fiber):
    """Creates a new product bundle object.

    Args:
      M: The manifold.
    """
    super().__init__(M, F)
    FiberBundle.__init__(self, M, F)

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return ProjectionMap(0, domain=self, image=self.manifold)

  def local_trivialization(self, x: Element) -> "ProductBundle":
    """Product bundles are globally trivial

    Args:
      p: A point on the base manifold

    Returns:
      A product bundle
    """
    return self

################################################################################################################

class TangentBundle(FiberBundle):
  """Tangent bundle of a manifold.  Represents union of
     tangent spaces at all points.  Elements are tuples
     of the form (p, v) where v is in TpM

  Attributes:
    M: The manifold.
  """
  Element = Tuple[Point,TangentVector]

  def __init__(self, M: Manifold):
    """Creates a new tangent space.

    Args:
      M: The manifold.
    """
    super().__init__(M, EuclideanManifold(dimension=M.dimension))

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return ProjectionMap(0, domain=self, image=self.manifold)

  def local_trivialization(self, p: Point) -> CartesianProductManifold:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi

    Args:
      p: A point on the base manifold

    Returns:
      (p, f), where p is in the base manifold and f is in the fiber
    """
    TpM = TangentSpace(p, self.manifold)
    return ProductBundle(self.manifold, TpM)

  def get_atlas(self):
    """Computations are done using local trivializations, so this
    shouldn't matter.
    """
    return None

################################################################################################################

class CotangentBundle(FiberBundle):
  """Cotangent bundle of a manifold.  Represents union of
     tangent spaces at all points.

  Attributes:
    M: The manifold.
  """
  Element = Tuple[Point,CotangentVector]

  def __init__(self, M: Manifold):
    """Creates a new tangent space.

    Args:
      M: The manifold.
    """
    super().__init__(M, EuclideanManifold(dimension=M.dimension))

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return ProjectionMap(0, domain=self, image=self.manifold)

  def local_trivialization(self, p: Point) -> CartesianProductManifold:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi

    Args:
      p: A point on the base manifold

    Returns:
      (p, f), where p is in the base manifold and f is in the fiber
    """
    coTpM = CotangentSpace(p, self.manifold)
    return ProductBundle(self.manifold, coTpM)

  def get_atlas(self):
    """Computations are done using local trivializations, so this
    shouldn't matter.
    """
    return None

################################################################################################################

class VectorBundle(FiberBundle, abc.ABC):
  """A fiber bundle where the fiber is R^k.  This is still an abstract
  class because we need a projection map and local trivialization.

  Attributes:
    F: The fiber F.
    M: The base space.
    E: The total space which looks looks like MxF around points.
    pi: Projection map from E to M.
  """
  Element = Tuple[Point,Coordinate]

  def __init__(self, M: Manifold, k: int):
    """Creates a new vector bundle object.

    Args:
      M: The manifold.
    """
    super().__init__(M, EuclideanManifold(dimension=k))

class FrameBundle(FiberBundle, abc.ABC):
  """Frame bundle.  Represents the space of frames that
  we can have over a manifold.

  Attributes:
    M: The manifold.
  """
  Element = Tuple[Point,Frame]

  def __init__(self, M: Manifold):
    """Creates a new frame bundle.

    Args:
      M: The manifold.
    """
    pass

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return ProjectionMap(0, domain=self, image=self.manifold)

  def local_trivialization(self, p: Point) -> CartesianProductManifold:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi

    Args:
      p: A point on the base manifold

    Returns:
      (p, f), where p is in the base manifold and f is in the fiber
    """
    TpM = TangentSpace(p, self.manifold)
    return ProductBundle(self.manifold, TpM)

  def get_atlas(self):
    """Computations are done using local trivializations, so this
    shouldn't matter.
    """
    return None

  # def right_action_map(self, g: GeneralLinearGroup, F: Frame) -> Map[Frame,Frame]:
  #   """The right action map of G on M.  This is set to
  #      a translation map by default.

  #   Args:
  #     g: An element of the Lie group
  #     M: The manifold to apply an action on

  #   Returns:
  #     theta_g
  #   """
  #   # HERE IS THE PART TO FIGURE OUT
  #   def theta_g(p: Frame) -> Frame:
  #     # Turn the frame into a matrix
  #     f = MatrixColsToList(TpM).inverse(F)

  #     return g.multiplication_map(p, g)
  #   return Map(theta_g, domain=M, image=M)

################################################################################################################

class CoFrameBundle(FiberBundle, abc.ABC):
  """Co-frame bundle.  Represents the space of coframes that
  we can have over a manifold.

  Attributes:
    M: The manifold.
  """
  Element = Tuple[Point,CoFrame]

  def __init__(self, M: Manifold):
    """Creates a new Coframe bundle.

    Args:
      M: The manifold.
    """
    pass

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return ProjectionMap(0, domain=self, image=self.manifold)

  def local_trivialization(self, p: Point) -> CartesianProductManifold:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi

    Args:
      p: A point on the base manifold

    Returns:
      (p, f), where p is in the base manifold and f is in the fiber
    """
    coTpM = CoTangentSpace(p, self.manifold)
    return ProductBundle(self.manifold, coTpM)

  def get_atlas(self):
    """Computations are done using local trivializations, so this
    shouldn't matter.
    """
    return None

  # def right_action_map(self, g: GeneralLinearGroup, F: CoFrame) -> Map[CoFrame,CoFrame]:
  #   """The right action map of G on M.  This is set to
  #      a translation map by default.

  #   Args:
  #     g: An element of the Lie group
  #     M: The manifold to apply an action on

  #   Returns:
  #     theta_g
  #   """
  #   # HERE IS THE PART TO FIGURE OUT
  #   def theta_g(p: CoFrame) -> CoFrame:
  #     # Turn the frame into a matrix
  #     f = MatrixColsToList(TpM).inverse(F)

  #     return g.multiplication_map(p, g)
  #   return Map(theta_g, domain=M, image=M)

################################################################################################################

class _GBundleMixin(abc.ABC):
  pass

################################################################################################################
