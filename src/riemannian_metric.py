from functools import partial
from typing import Callable, List, Optional, Union, Tuple
from collections import namedtuple
import src.util
from functools import partial
import jax
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.tangent import *
from src.cotangent import *
from src.vector import *
from src.section import *
from src.bundle import *
from src.tensor import *
from src.instances.manifolds import EuclideanManifold
import src.util as util
import einops
import itertools
import abc
import math

__all__ = ["RiemannianMetric",
           "EuclideanMetric",
           "RiemannianManifold",
           "TangentToCotangetBundleIsomorphism",
           "CotangentToTangetBundleIsomorphism",
           "gradient_vector_field"]

class RiemannianMetric(SymmetrizedTensorField):
  """A riemannian metric is a symmetric covariant 2-tensor field

  Attributes:
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, M: Manifold):
    """Create a new Riemannian metric

    Args:
      M: The base manifold.
    """
    # A riemannian metric is a symmetric covariant 2-tensor field on M
    self.type = TensorType(0, 2)
    self.manifold = M
    super().__init__(self.type, self.manifold)

################################################################################################################

class EuclideanMetric(RiemannianMetric):
  """The Euclidean metric is delta_{ij}dx^i dx^j

  Attributes:
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, dimension: int, coordinate_function: Optional[Callable[[Point,bool],Coordinate]]=None):
    """Create Euclidean space

    Args:
      dimension: Dimension
      coordinate_function: Optionally give a prefered choice of coordinates
    """
    M = EuclideanManifold(dimension, chart=coordinate_function)
    super().__init__(M)

  @property
  def T(self) -> TensorField:
    """Don't actually need this
    """
    assert 0, "Shouldn't be calling this"

  def apply_to_co_vector_fields(self, *Xs: List[Union[VectorField,CovectorField]]) -> Map:
    """Evaluate the tensor field on (co)vector fields

    Args:
      Xs: A list of (co)vector fields

    Returns:
      A map over the manifold
    """
    def fun(p: Point):
      return self(p)(*[X(p) for X in Xs])
    return Map(fun, domain=self.manifold, image=Reals())

  def apply_to_point(self, p: Point) -> Tensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    # The coordinates are delta_{ij}
    coords = jnp.eye(self.manifold.dimension)

    # Construct the tangent covector
    TkTpM = TensorSpace(p, self.type, self.manifold)
    return Tensor(coords, TkTpM=TkTpM)

################################################################################################################

class RiemannianManifold(Manifold):
  """A Riemannian manifold is a smooth manifold with a Riemannian metric

  Attributes:
    g: A Riemannian metric
    atlas: Atlas providing coordinate representation.
  """
  def __init__(self, g: RiemannianMetric, dimension: int=None):
    """Create a Riemannian manifold object.

    Args:
      g: A Riemannian metric
      dimension: Dimension of the manifold
    """
    assert isinstance(g, RiemannianMetric)
    self.g = g
    super().__init__(dimension=dimension)

  def inner_product(self, Xp: TangentVector, Yp: TangentVector) -> Coordinate:
    """Evaluate the inner product between tangent vectors at p.

    Args:
      Xp: A tangent vector
      Yp: A tangent vector

    Returns:
      <Xp,Yp>_g
    """
    gp = self.g(X.p)
    return gp(Xp, Yp)

################################################################################################################

class TangentToCotangetBundleIsomorphism(InvertibleMap[TangentBundle,CotangentBundle]):
  """Given a Riemannian metric g on M, there is a bundle homomorphism
  from g_hat: TM -> TM* defined by g_hat(v)(w) = g_p(v, w) where
  p in M, v in TpM, g_hat(v) in TpM*

  Attributes:
    g: The Riemannian metric
  """
  def __init__(self, g: RiemannianMetric):
    """Create a bundle isomorphism

    Attributes:
      g: Riemannian metric
    """
    self.g = g

  def apply_to_vector(self, Xp: TangentVector) -> CotangentVector:
    """Converts a tangent vector to a cotangent vector using the Riemannian metric.
    if g = g_{ij}dx^i dx^j, X = X^i d/dx^i, Y = Y^j d/dx^j, then g_hat(X) = g_{ij}X^i dx^j
    This is lowering an index.

    Args:
      Xp: A tangent vector

    Returns:
      g_hat(Xp)
    """
    assert isinstance(Xp, TangentVector)
    p = Xp.p

    # Get the metric at p
    gp = self.g(p)

    # Get the coordinates for g in terms of the coordinate function for Xp
    g_coords = gp.get_coordinates(Xp.phi)

    # Get the coordinates of the new
    g_hatX_coords = jnp.einsum("ij,i->j", g_coords, Xp.x)

    # Form the dual space
    coTpM = Xp.TpM.get_dual_space()

    return CotangentVector(g_hatX_coords, coTpM=coTpM)

  def apply_to_vector_field(self, X: VectorField) -> CovectorField:
    """Turn a vector field into a covector field by lowering an index.

    Args:
      X: A vector field

    Returns:
      g_hat(X)
    """
    class LowerdIndexCovectorField(CovectorField):
      def __init__(self, isomorphism: "TangentToCotangetBundleIsomorphism", X: VectorField):
        self.X = X
        self.isomorphism = isomorphism
        super().__init__(self.X.manifold)

      def apply_to_point(self, p: Point) -> CotangentVector:
        Xp = self.X(p)
        return self.isomorphism.apply_to_vector(Xp)

    return LowerdIndexCovectorField(self, X)

  def __call__(self, v: Union[TangentVector,VectorField]) -> Union[CotangentVector,CovectorField]:
    if isinstance(v, VectorField):
      return self.apply_to_vector_field(v)
    assert isinstance(v, TangentVector)
    return self.apply_to_vector(v)

  def get_inverse(self):
    """Creates an inverse of the function

    Returns:
      A new Function object that is the inverse of this one.
    """
    return CotangentToTangetBundleIsomorphism(self.g)

  def inverse(self, q: Output) -> Input:
    """Applies the inverse function on q.

    Args:
      q: An input coordinate.

    Returns:
      f^{-1}(q)
    """
    return self.get_inverse()(q)

################################################################################################################

class CotangentToTangetBundleIsomorphism(InvertibleMap[TangentBundle,CotangentBundle]):
  """Inverse of TangentToCotangetBundleIsomorphism

  Attributes:
    g: The Riemannian metric
  """
  def __init__(self, g: RiemannianMetric):
    """Create a bundle isomorphism

    Attributes:
      g: Riemannian metric
    """
    self.g = g

  def apply_to_covector(self, wp: CotangentVector) -> TangentVector:
    """Converts a tangent vector to a cotangent vector using the Riemannian metric.
    if g = g_{ij}dx^i dx^j, X = X^i d/dx^i, Y = Y^j d/dx^j, then g_hat(X) = g_{ij}X^i dx^j
    This is lowering an index.

    Args:
      wp: A tangent vector

    Returns:
      g_hat(wp)
    """
    assert isinstance(wp, CotangentVector)
    p = wp.p

    # Get the metric at p
    gp = self.g(p)

    # Get the coordinates for g in terms of the coordinate function for wp
    g_coords = gp.get_coordinates(wp.phi)
    g_inv_coords = jnp.linalg.inv(g_coords)

    # Get the coordinates of the new
    g_inv_hatX_coords = jnp.einsum("ij,j->i", g_inv_coords, wp.x)

    # Form the dual space
    TpM = wp.coTpM.get_dual_space()

    return TangentVector(g_inv_hatX_coords, TpM=TpM)

  def apply_to_covector_field(self, w: CovectorField) -> VectorField:
    """Turn a vector field into a covector field by lowering an index.

    Args:
      w: A vector field

    Returns:
      g_hat(w)
    """
    class RaisedIndexVectorField(VectorField):
      def __init__(self, isomorphism: "CotangentToTangetBundleIsomorphism", w: CovectorField):
        self.w = w
        self.isomorphism = isomorphism
        super().__init__(self.w.manifold)

      def apply_to_point(self, p: Point) -> TangentVector:
        Xp = self.w(p)
        return self.isomorphism.apply_to_covector(Xp)

    return RaisedIndexVectorField(self, w)

  def __call__(self, w: Union[CotangentVector,CovectorField]) -> Union[TangentVector,VectorField]:
    if isinstance(w, CovectorField):
      return self.apply_to_covector_field(w)
    assert isinstance(w, CotangentVector)
    return self.apply_to_covector(w)

  def get_inverse(self):
    """Creates an inverse of the function

    Returns:
      A new Function object that is the inverse of this one.
    """
    return TangentToCotangetBundleIsomorphism(self.g)

  def inverse(self, q: Output) -> Input:
    """Applies the inverse function on q.

    Args:
      q: An input coordinate.

    Returns:
      f^{-1}(q)
    """
    return self.get_inverse()(q)

################################################################################################################

def gradient_vector_field(f: Map[Point,Coordinate], g: RiemannianMetric):
  df = FunctionDifferential(f)
  return CotangentToTangetBundleIsomorphism(g).apply_to_covector_field(df)