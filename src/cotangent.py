from functools import partial
from typing import Callable, List, Optional, Union
import src.util
from functools import partial
import jax
import abc
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.tangent import *
from src.vector import *
import src.util as util

__all__ = ["CotangentVector",
           "CotangentSpace",
           "CotangentBasis",
           "CovectorField",
           "Coframe",
           "PullbackCovectorField",
           "FunctionDifferential",
           "Pullback",
           "pullback"]

class CotangentVector(Vector, LinearMap[TangentVector,Coordinate]):
  """A covector on a Manifold

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    coTpM: The tangent space that the vector lives on.
  """
  def __init__(self, x: Coordinate, coTpM: "CotangentSpace"):
    """Creates a new covector.

    Args:
      x: The coordinates of the covector in the basis induced by a chart.
      coTpM: The cotangent space that the covector lives on.
    """
    super().__init__(x, coTpM)
    self.coTpM = coTpM

    # Get the chart from the vector space
    self.phi = self.V.phi
    self.phi_inv = self.phi.get_inverse()

    # Also keep track of the point that the tangent space is defined on
    # and its coordinates
    self.p = coTpM.p
    self.p_hat = self.phi(self.p)

  def __call__(self, v: TangentVector) -> Coordinate:
    """Apply the covector to a tangent vector v

    Args:
      v: Input tangent vector

    Returns:
      w(v)
    """
    assert isinstance(v, TangentVector)
    # assert v.TpM.get_dual_space() == self.coTpM

    # Evaluate the vector components in this basis
    v_coords = v.get_coordinates(self.phi)
    return jnp.vdot(self.x, v_coords)

  def get_coordinates(self, component_function: Chart) -> Coordinate:
    """Get the coordinates of this vector in terms of coordinates function

    Args:
      component_function: A chart that gives coordinates

    Returns:
      Coordinates of this vector for component_function
    """
    # This is the inverse of the map from x_coords to component_function_coords
    psiinv_phi = compose(self.phi, component_function.get_inverse())
    q = component_function(self.p)
    _, vjp = jax.vjp(psiinv_phi.f, q)
    coords, = vjp(self.x)
    return coords

################################################################################################################

class CotangentSpace(VectorSpace):
  """Cotangent space of a manifold.  Is defined at a point, T_p^*M

  Attributes:
    p: The point where the space lives.
    M: The manifold.
  """
  Element = CotangentVector

  def __init__(self, p: Point, M: Manifold):
    """Creates a new cotangent space.

    Args:
      p: The point where the space lives.
      M: The manifold.
    """
    self.p = p
    self.manifold = M

    # Keep track of the chart function for the manifold
    self.phi = self.manifold.get_chart_for_point(self.p)
    super().__init__(dimension=self.manifold.dimension)

  def __eq__(self, other: "CotangentSpace") -> bool:
    """Checks to see if 2 cotangent spaces are equal.

    Args:
      other: The other cotangent space.

    Returns:
      True if they are the same, False otherwise.
    """
    point_same = True
    if util.GLOBAL_CHECK:
      point_same = jnp.allclose(self.p, other.p)
    return point_same and (self.manifold == other.manifold)

  def __contains__(self, v: CotangentVector) -> bool:
    """Checks to see if p exists in the manifold.  This is the case if
       the point is in the domain of any chart

    Args:
      p: Test point.

    Returns:
      True if p is in the manifold, False otherwise.
    """
    type_check = isinstance(v, CotangentVector)
    val_check = v.coTpM == self
    return type_check and val_check

  def get_dual_space(self) -> TangentSpace:
    """Get the dual space corresponding to the cotangent space

    Returns:
      Tangent space
    """
    from src.tangent import TangentSpace
    return TangentSpace(self.p, self.manifold)

################################################################################################################

class CotangentBasis(VectorSpaceBasis):
  """A list of cotangent vectors that forms a basis for the cotangent space.

  Attributes:
    basis: A list of cotangent vectors
    coTpM: The cotangent space that the vector lives on.
  """
  def __init__(self, basis_vectors: List[CotangentVector], coTpM: CotangentSpace):
    """Create a new cotangent space basis object

    Args:
      basis_vectors: A list of basis vectors
      coTpM: The vector space that this is a basis of
    """
    super().__init__(basis_vectors, coTpM)
    self.coTpM = coTpM

  @staticmethod
  def from_tangent_basis(tangent_basis: TangentBasis) -> "CotangentBasis":
    """Construct the dual basis of a basis for the tangent space.

    Args:
      tangent_basis: A basis for the tangent space.

    Returns:
      The corresponding cotangent space.
    """
    from src.bundle import FrameBundle, CoframeBundle

    TpM = tangent_basis.TpM
    coTpM = TpM.get_dual_space()

    # First use a local trivialization of the tangent bundle at the basis
    # in order to get the matrix representation of the basis
    frame_bundle = FrameBundle(TpM.manifold)
    lt_map = frame_bundle.get_local_trivialization_map(tangent_basis)
    p, matrix = lt_map(tangent_basis)

    # Invert
    inv_matrix = jnp.linalg.inv(matrix)

    # Now use a local trivialization of the coframe bundle to go to
    # the cotangent basis
    coframe_bundle = CoframeBundle(TpM.manifold)
    co_lt_map = coframe_bundle.get_local_trivialization_map(p) # TODO: Whats the right thing to pass in here?
    return co_lt_map.inverse((p, inv_matrix))

################################################################################################################

from src.section import Section
class CovectorField(Section[Point,CotangentVector], abc.ABC):
  """A covector field is a smooth section of the projection map from the
  cotangent bundle to the base manifold.  Takes a point and outputs
  a tangent vector on the tangent space at the point.  Treat this
  as a section of the tangent bundle, so will need to define s(p) in R^n
  in order to use.

  Attributes:
    X: Function that assigns a tangent vector to every point on the manifold
    M: The manifold that the vector field is defined on
  """
  def __init__(self, M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      M: The base manifold.
    """
    self.manifold = M

    from src.tensor import TensorType
    self.type = TensorType(0, 1)

    # Construct the bundle
    domain = M
    from src.bundle import CotangentBundle
    image = CotangentBundle(M)
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    super().__init__(pi)

  def apply_to_vector_field(self, X: VectorField) -> Map:
    """Fill the covector field with a vector field.  This
    produces a map over tha manifold

    Args:
      X: A vector field

    Returns:
      w(X)
    """
    assert isinstance(X, VectorField)
    def fun(p: Point):
      return self(p)(X(p))
    return Map(fun, domain=self.manifold, image=Reals())

  @abc.abstractmethod
  def apply_to_point(self, p: Point) -> CotangentVector:
    """Evaluate the covector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    pass

  def __call__(self, x: Union[Point,VectorField]) -> Union[CotangentVector,Map]:
    """Evaluate the covector field at a point.

    Args:
      x: Point on the manifold.

    Returns:
      Cotangent vector at x.
    """
    if isinstance(x, VectorField):
      return self.apply_to_vector_field(x)
    else:
      if util.GLOBAL_CHECK:
        assert x in self.manifold
      return self.apply_to_point(x)

################################################################################################################

class PullbackCovectorField(CovectorField):
  """The pullback of a covector field w through a map F

  Attributes:
    F: Map
    w: Covector field
  """
  def __init__(self, F: Map, w: CovectorField):
    """Create a new pullback covector field object

    Args:
      F: Map
      w: Covector field
    """
    assert w.manifold == F.image
    self.F = F
    self.w = w
    super().__init__(M=F.domain)

  def apply_to_point(self, p: Point) -> CotangentVector:
    """Evaluate the covector field at a point.

    Args:
      q: Point on the manifold.

    Returns:
      Tangent vector at q.
    """
    wp = self.w(self.F(p))
    dFp_ = self.F.get_pullback(p)
    return dFp_(wp)

def pullback(F: Map, w: VectorField) -> PullbackCovectorField:
  """The pullback of w by F.  F^*(w)

  Args:
    F: A map
    w: A covector field on defined on the image of F

  Returns:
    F^*(w)
  """
  return PullbackCovectorField(F, w)

################################################################################################################

class FunctionDifferential(CovectorField):
  """The differential of a real valued function

  Attributes:
    f: The real valued function
    M: The manifold that the vector field is defined on
  """
  def __init__(self, f: Map[Point,Coordinate]):
    assert (f.image.dimension is None) or (f.image.dimension <= 1)
    assert isinstance(f.image, Reals)
    self.function = f
    M = self.function.domain
    super().__init__(M)

  def apply_to_point(self, p: Point) -> CotangentVector:
    """The differential at p is a covector so that df_p(v) = v_p(f)

    Args:
      p: Point on the manifold.

    Returns:
      Differential at p.
    """
    # p is in the domain of f

    # Get the coordinate map for f at p
    f_hat = self.function.get_coordinate_map(p)

    # Get the coordinates for p
    phi = self.domain.get_chart_for_point(p)
    p_hat = phi(p)

    # The coordinates change with a vjp
    out, vjp = jax.vjp(f_hat.f, p_hat)
    z, = vjp(jnp.ones_like(out))
    # z = jax.grad(f_hat.f)(p_hat)

    coTpM = CotangentSpace(p, self.domain)
    return CotangentVector(z, coTpM)

################################################################################################################

class Pullback(LinearMap[CotangentVector,CotangentVector]):
  """The pullback map for a function

  Attributes:
    F: A map from M->N.
    p: The point where the differential is defined.
  """
  def __init__(self, F: Map[Input,Output], p: Point):
    """Creates the differential of F at p

    Args:
    F: A map from M->N.
    p: The point where the differential is defined.
    """
    self.p = p
    self.q = F(self.p)
    self.F = F

    self.coTpM = CotangentSpace(self.p, M=F.domain)
    self.coTpN = CotangentSpace(self.q, M=F.image)
    super().__init__(f=self.__call__, domain=self.coTpN, image=self.coTpM)

  def __call__(self, w: CotangentVector) -> CotangentVector:
    """Apply the differential to a tangent vector.

    Args:
      w: A cotangent vector on N = F(M)

    Returns:
      dF_p^*(w)
    """
    assert isinstance(w, CotangentVector)

    # Need to use the coordinate map to get the coordinate change.
    F_hat = self.F.get_coordinate_map(self.p)

    phi = self.coTpM.manifold.get_chart_for_point(self.p)
    p_hat = phi(self.p)

    # The coordinates change with a vjp
    q_hat, vjp = jax.vjp(F_hat.f, p_hat)
    z, = vjp(w.x)

    coTpM = CotangentSpace(self.p, self.F.domain)
    return CotangentVector(z, coTpM)

  def get_coordinates(self) -> Coordinate:
    """Return the coordinate representation of this map

    Args:
      None

    Returns:
      The coordinates
    """
    # Get the coordinate map function
    cm = self.F.get_coordinate_map(self.p)

    # Find the coordinate representation of the point
    chart = self.F.domain.get_chart_for_point(self.p)
    p_coords = chart(self.p)

    # Get the Jacobian of the transformation
    return jax.jacobian(cm)(p_coords).T

################################################################################################################

class Coframe(Section[Point,CotangentBasis], abc.ABC):
  """A coframe is a collection of linearly independent covector fields
  that form a basis for the cotangent space.

  Attributes:
    M: Manifold
  """
  def __init__(self, M: Manifold):
    """Creates a new frame

    Args:
      M: Manifold
    """
    self.manifold = M

    domain = M
    from src.bundle import FrameBundle
    self.coframe_bundle = CoFrameBundle(M)
    pi = ProjectionMap(idx=0, domain=self.coframe_bundle, image=domain)
    super().__init__(pi)

  @abc.abstractmethod
  def apply_to_point(self, p: Input) -> CotangentBasis:
    """Evaluate the coframe field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Cotangent basis at p
    """
    pass
