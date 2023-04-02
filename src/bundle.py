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
from src.instances.manifolds import EuclideanManifold
from src.vector import *
from src.instances.lie_groups import GeneralLinearGroup
import diffrax
import abc

__all__ = ["FiberBundle",
           "ProductBundle",
           "TangentBundle",
           "GlobalDifferential",
           "CotangentBundle",
           "FrameBundle",
           "CoframeBundle"]

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
  def get_local_trivialization_map(self, x: Element) -> Map["FiberBundle","ProductBundle"]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    pass

  @abc.abstractmethod
  def __contains__(self, x: Element) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    pass

  def get_chart_for_point(self, x: Element) -> Chart:
    """Get a chart to use at point x in the total space.  Do this by
    getting a chart of the local trivialization at pi(x)

    Args:
      The input point

    Returns:
      The chart that contains x in its domain
    """
    lt = self.get_local_trivialization_map(x)
    UxF = lt.image
    lt_chart = UxF.get_chart_for_point(lt(x))
    chart = compose(lt_chart, lt)
    return chart

################################################################################################################

class ProductBundle(CartesianProductManifold,FiberBundle):
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
    self.manifold, self.fiber = M, F
    super().__init__(M, F)
    FiberBundle.__init__(self, M, F)

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return ProjectionMap(0, domain=self, image=self.manifold)

  def get_local_trivialization_map(self, x: Element) -> Map["ProductBundle","ProductBundle"]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    image = ProductBundle(self.manifold, self.fiber)
    def phi(x, inverse=False):
      return x
    return Diffeomorphism(x, domain=self, image=image)

  def __contains__(self, x: Element) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    p, v = x
    return (p in self.manifold) and (v in self.fiber)

################################################################################################################

class TangentBundle(FiberBundle):
  """Tangent bundle of a manifold.  Represents union of
     tangent spaces at all points.  Elements are tuples
     of the form (p, v) where v is in TpM

  Attributes:
    M: The manifold.
  """
  Element = TangentVector # The tangent vector is evaluated at p!

  def __init__(self, M: Manifold):
    """Creates a new tangent space.

    Args:
      M: The manifold.
    """
    super().__init__(M, EuclideanManifold(dimension=M.dimension))

  def get_projection_map(self) -> Map[TangentVector,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda v: v.p, domain=self, image=self.manifold)

  def get_local_trivialization_map(self, x: TangentVector) -> Map[TangentVector,Tuple[Point,Coordinate]]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    assert isinstance(x, TangentVector)
    image = ProductBundle(self.manifold, EuclideanManifold(dimension=x.TpM.dimension))
    def Phi(v, inverse=False):
      if inverse == False:
        return v.p, v.x
      else:
        p, x = v
        return TangentVector(x, TangentSpace(p, self.manifold))
    return Diffeomorphism(Phi, domain=self, image=image)

  def get_atlas(self):
    """Computations are done using local trivializations, so this
    shouldn't matter.
    """
    return None

  def __contains__(self, v: TangentVector) -> bool:
    """Checks to see if v exists in the bundle.

    Args:
      v: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    return (v.p in self.manifold) and (v.x in self.fiber)

class GlobalDifferential(LinearMap[TangentBundle,TangentBundle]):
  """The global differential is the union of all of the
  differentials at every point on the manifold
  """
  def __init__(self, F: Map, M: Manifold):
    """Creates a new global differential
    Args:
      M: The manifold.
    """
    self.F = F
    self.manifold = M
    self.domain = TangentBundle(M)
    self.image = TangentBundle(apply_map_to_manifold(self.F, self.manifold))
    super().__init__(self.__call__, domain=self.domain, image=self.image)

  def __call__(self, x: TangentVector) -> TangentVector:
    """Compute the differential of a tangent vector

    Args:
      x: A tangent vector on the tangent bundle

    Returns:
      The differential of F applied to x
    """
    assert isinstance(x, TangentVector)
    return self.F.get_differential(x.p)(x)

################################################################################################################

class CotangentBundle(FiberBundle):
  """Cotangent bundle of a manifold.  Represents union of
     tangent spaces at all points.

  Attributes:
    M: The manifold.
  """
  Element = CotangentVector

  def __init__(self, M: Manifold):
    """Creates a new tangent space.

    Args:
      M: The manifold.
    """
    super().__init__(M, EuclideanManifold(dimension=M.dimension))

  def get_projection_map(self) -> Map[CotangentVector,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda v: v.p, domain=self, image=self.manifold)

  def get_local_trivialization_map(self, x: Element) -> Map[CotangentVector,Tuple[Point,Coordinate]]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    assert isinstance(x, CotangentVector)
    image = ProductBundle(self.manifold, EuclideanManifold(dimension=x.TpM.dimension))
    def Phi(v, inverse=False):
      if inverse == False:
        return v.p, v.x
      else:
        p, x = v
        return CotangentVector(x, CotangentSpace(p, self.manifold))
    return Diffeomorphism(Phi, domain=self, image=image)

  def get_atlas(self):
    """Computations are done using local trivializations, so this
    shouldn't matter.
    """
    return None

  def __contains__(self, w: CotangentVector) -> bool:
    """Checks to see if w exists in the bundle.

    Args:
      w: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    return (w.p in self.manifold) and (w.x in self.fiber)

################################################################################################################

class FrameBundle(FiberBundle, abc.ABC):
  """Frame bundle.  Represents the space of frames that
  we can have over a manifold.

  Attributes:
    M: The manifold.
  """
  Element = TangentBasis # Every element is a basis for the tangent space

  def __init__(self, M: Manifold):
    """Creates a new frame bundle.

    Args:
      M: The manifold.
    """
    super().__init__(M, GeneralLinearGroup(dim=M.dimension))

  def __contains__(self, x: TangentBasis) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    return x.TpM.manifold == self.manifold

  def get_projection_map(self) -> Map[TangentBasis,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda x: x.TpM.p, domain=self, image=self.manifold)

  def get_local_trivialization_map(self, x: TangentBasis) -> Map[TangentBasis,Tuple[Point,Coordinate]]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    # assert isinstance(x, TangentBasis)

    def Phi(inpt, inverse=False):
      if inverse == False:
        tangent_basis = inpt
        p = tangent_basis.TpM.p
        mat = jnp.stack([v.x for v in tangent_basis.basis], axis=1)
        return p, mat
      else:
        p, mat = inpt
        xs = jnp.split(mat, self.manifold.dimension, axis=1) # Split on cols

        # Need to recreate this so that we can pass gradients through p!
        TpM = TangentSpace(p, self.manifold)
        basis = TangentBasis([TangentVector(x.ravel(), TpM) for x in xs], TpM)
        return basis

    image = ProductBundle(self.manifold, GeneralLinearGroup(dim=self.manifold.dimension))
    return Diffeomorphism(Phi, domain=self, image=image)

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
  #     f = MatrixColsToTangentBasis(TpM).inverse(F)

  #     return g.multiplication_map(p, g)
  #   return Map(theta_g, domain=M, image=M)

################################################################################################################

class CoframeBundle(FiberBundle, abc.ABC):
  """Co-frame bundle.  Represents the space of coframes that
  we can have over a manifold.

  Attributes:
    M: The manifold.
  """
  Element = CotangentBasis

  def __init__(self, M: Manifold):
    """Creates a new Coframe bundle.

    Args:
      M: The manifold.
    """
    super().__init__(M, GeneralLinearGroup(dim=M.dimension))

  def __contains__(self, x: CotangentBasis) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    return x.coTpM.manifold == self.manifold

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda x: x.TpM.p, domain=self, image=self.manifold)

  def get_local_trivialization_map(self, x: CotangentBasis) -> Map[CotangentBasis,Tuple[Point,Coordinate]]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    # assert isinstance(x, CotangentBasis)

    def Phi(inpt, inverse=False):
      if inverse == False:
        cotangent_basis = inpt
        p = cotangent_basis.TpM.p
        mat = jnp.stack([v.x for v in cotangent_basis.basis], axis=0)
        return p, mat
      else:
        p, mat = inpt
        xs = jnp.split(mat, self.manifold.dimension, axis=0) # Split on rows

        # Need to recreate this so that we can pass gradients through p!
        coTpM = CotangentSpace(p, self.manifold)
        basis = CotangentBasis([CotangentVector(x.ravel(), coTpM) for x in xs], coTpM)
        return basis

    image = ProductBundle(self.manifold, GeneralLinearGroup(dim=self.manifold.dimension))
    return Diffeomorphism(Phi, domain=self, image=image)

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
  #     f = MatrixColsToTangentBasis(TpM).inverse(F)

  #     return g.multiplication_map(p, g)
  #   return Map(theta_g, domain=M, image=M)

################################################################################################################

class _GBundleMixin(abc.ABC):
  pass

################################################################################################################
