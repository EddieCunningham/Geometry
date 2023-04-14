from functools import partial
from typing import Callable, List, Optional, Generic, Tuple, Union
import src.util
from functools import partial
import jax
import abc
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.vector import *
import src.util as util

__all__ = ["TangentVector",
           "TangentSpace",
           "TangentBasis",
           "VectorField",
           "CoordinateVectorField",
           "Frame",
           "StandardCoordinateFrame",
           "Differential"]

class TangentVector(Vector):
  """A tangent vector on a Manifold

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    TpM: The tangent space that the vector lives on.
    phi: A chart for the manifold at p.  This the coordinate function
         that the "x" coordinates correspond to.
  """
  def __init__(self, x: Coordinate, TpM: "TangentSpace"):
    """Creates a new tangent vector.

    Args:
      x: The coordinates of the vector in the basis induced by a chart.
      TpM: The tangent space that the vector lives on.
    """
    assert x.shape[-1] == TpM.dimension
    super().__init__(x, TpM)
    self.TpM = TpM

    # Get the chart from the vector space
    self.phi = self.V.phi
    self.phi_inv = self.phi.get_inverse()

    # Also keep track of the point that the tangent space is defined on
    # and its coordinates
    self.p = self.TpM.p

  @property
  def p_hat(self):
    if hasattr(self, "_p_hat") == False:
      self._p_hat = self.phi(self.p)
    return self._p_hat

  def __call__(self, f: Function) -> Coordinate:
    """Apply the tangent vector to f.

    Args:
      f: An input function

    Returns:
      V(f)
    """
    assert isinstance(f, Map)
    f_phiinv = compose(f, self.phi_inv)
    return jax.jvp(f_phiinv.f, (self.p_hat,), (self.x,))[1]

  def get_coordinates(self, component_function: Chart) -> Coordinate:
    """Get the coordinates of this vector in terms of coordinates function

    Args:
      component_function: A chart that gives coordinates

    Returns:
      Coordinates of this vector for component_function
    """
    return self(component_function)

################################################################################################################

class TangentSpace(VectorSpace):
  """Tangent space of a manifold.  Is defined at a point, T_pM

  Attributes:
    p: The point where the space lives.
    M: The manifold.
  """
  Element = TangentVector

  def __init__(self, p: Element, M: Manifold):
    """Creates a new tangent space.

    Args:
      p: The point where the space lives.
      M: The manifold.
    """
    self.p = p
    self.manifold = M

    # Keep track of the chart function for the manifold
    self.phi = self.manifold.get_chart_for_point(self.p)
    super().__init__(dimension=self.manifold.dimension)

  def __eq__(self, other: "TangentSpace") -> bool:
    """Checks to see if 2 tangent spaces are equal.

    Args:
      other: The other tangent space.

    Returns:
      True if they are the same, False otherwise.
    """
    point_same = True
    if util.GLOBAL_CHECK:
      point_same = jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, self.p, other.p))
    return point_same and (self.manifold == other.manifold)

  def get_dual_space(self) -> "CotangentSpace":
    """Get the dual space corresponding to the tangent space

    Returns:
      Cotangent space
    """
    from src.cotangent import CotangentSpace
    return CotangentSpace(self.p, self.manifold)

  def get_basis(self) -> "TangentBasis":
    """Get a basis of vectors for the vector space

    Returns:
      A list of vector that form a basis for the vector space
    """
    eye = jnp.eye(self.dimension)
    basis = []
    for i in range(self.dimension):
      v = TangentVector(eye[i], self)
      basis.append(v)
    return TangentBasis(basis, self)

  def get_dual_basis(self) -> "CotangentBasis":
    """Get a basis for the dual space

    Returns:
      Cotangent space basis
    """
    return self.get_dual_space().get_basis()

################################################################################################################

from src.section import Section
class VectorField(Section[Point,TangentVector], abc.ABC):
  """A vector field is a smooth section of the projection map from the
  tangent bundle to the base manifold.  Takes a point and outputs
  a tangent vector on the tangent space at the point.  Treat this
  as a section of the tangent bundle, so will need to define s(p) in R^n
  in order to use.

  Attributes:
    M: The manifold that the vector field is defined on
  """
  def __init__(self, M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      M: The base manifold.
    """
    self.manifold = M

    # Construct the bundle
    domain = M
    from src.bundle import TangentBundle
    image = TangentBundle(M)
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    super().__init__(pi)

  def apply_to_map(self, f: Map) -> Map:
    """Multiply a vector field with a function.

    Args:
      f: A map from the manifold to R

    Returns:
      Xf
    """
    from src.instances.manifolds import EuclideanManifold
    assert isinstance(f, Map)
    def fun(p: Point):
      return self(p)(f)
    return Map(fun, domain=self.manifold, image=EuclideanManifold(dimension=1))

  @abc.abstractmethod
  def apply_to_point(self, p: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    pass

  def __call__(self, x: Union[Point,Map]) -> Union[TangentVector,Map]:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    if isinstance(x, Map):
      return self.apply_to_map(x)
    else:
      if util.GLOBAL_CHECK:
        assert x in self.manifold
      return self.apply_to_point(x)

  def get_coordinate_functions_with_basis(self) -> List[Tuple[Map[Manifold,Coordinate],"CoordinateVectorField"]]:
    """Get the coordinate functions associated with basis elements

    Returns:
      A list of coordinate functions
    """
    # Get the coordinate (co)frame over the manifold
    coordinate_frame = StandardCoordinateFrame(self.manifold)
    basis_vector_fields = coordinate_frame.to_vector_field_list()

    dual_frame = coordinate_frame.get_dual_coframe()
    basis_covector_fields = dual_frame.to_covector_field_list()

    out = []
    for e, E in zip(basis_covector_fields, basis_vector_fields):
      # Get the coordinate function
      c = e(self)
      out.append((c, E))

    return out

class CoordinateVectorField(VectorField):
  """d/dx^i

  Attributes:
    i: The coordinate index
    M: The manifold that the vector field is defined on
  """
  def __init__(self, i: int, M: Manifold):
    """Create a coordinate vector field

    Attributes:
      i: The coordinate index
      M: The manifold that the vector field is defined on
    """
    self.i = i
    super().__init__(M)

  def apply_to_point(self, p: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    TpM = TangentSpace(p, self.manifold)
    coords = jnp.zeros(self.manifold.dimension)
    coords = coords.at[self.i].set(1)
    return TangentVector(coords, TpM=TpM)

################################################################################################################

class TangentBasis(VectorSpaceBasis):
  """A list of tangent vectors that forms a basis for the tangent space.

  Attributes:
    basis: A list of tangent vectors
    TpM: The tangent space that the vector lives on.
  """
  def __init__(self, basis_vectors: List[TangentVector], TpM: TangentSpace):
    """Create a new cotangent space basis object

    Args:
      basis_vectors: A list of basis vectors
      TpM: The vector space that this is a basis of
    """
    super().__init__(basis_vectors, TpM)
    self.TpM = TpM

  def __call__(self, v: Coordinate) -> TangentVector:
    """Turns a vector in R^n into a vector on TpM

    Args:
      v: A vector in Rn

    Returns:
      A tangent vector
    """
    assert v in Reals(dimension=self.TpM.manifold.dimension)

    mat = jnp.stack([v.x for v in self.basis], axis=1)
    new_v = mat@v

    return TangentVector(new_v, self.TpM)

  def inverse(self, v: TangentVector) -> Coordinate:
    """Get the Euclidean coordinates of a tangent vector.
    This is the inverse of __call__

    Args:
      v: A tangent vector

    Returns:
      Coordianates
    """
    mat = jnp.stack([v.x for v in self.basis], axis=1)
    mat_inv = jnp.linalg.inv(mat)
    return mat_inv@v.x

  def get_inverse(self) -> Diffeomorphism:
    """Get the inverse function as a Diffeomorphism object

    Returns:
      Diffeomorphism
    """
    return Diffeomorphism(self.inverse, domain=self.TpM, image=EuclideanManifold(dimension=self.TpM.manfiold.dimension))

  def call_unvectorized(self, v: Coordinate) -> TangentVector:
    """Turns a vector in R^n into a vector on TpM

    Args:
      v: A vector in Rn

    Returns:
      A tangent vector
    """
    assert v in Reals(dimension=self.TpM.manifold.dimension)

    ans = v[0]*self.basis[0]
    for i, (vi, ei) in enumerate(zip(v, self.basis)):
      if i == 0:
        continue

      ans += vi*ei

    return ans

################################################################################################################

class Frame(Section[Point,TangentBasis], abc.ABC):
  """A frame is a collection of linearly independent vector fields
  that form a basis for the tangent space.  Treat this as a section
  of the frame bundle, so will need to define s(p) in GL(n,R) in order
  to use.

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
    self.frame_bundle = FrameBundle(M)
    pi = ProjectionMap(idx=0, domain=self.frame_bundle, image=domain)
    super().__init__(pi)

  @abc.abstractmethod
  def apply_to_point(self, p: Point) -> TangentBasis:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    pass

  def get_dual_coframe(self) -> "Coframe":
    """Return the coframe associated with this frame

    Returns:
      Dual coframe
    """
    from src.cotangent import Coframe, CotangentBasis
    class DualCoframe(Coframe):
      def __init__(self, frame):
        self.frame = frame
        super().__init__(frame.manifold)

      def apply_to_point(self, p: Input) -> CotangentBasis:
        # Get the local trivialization maps
        self.frame_lt_map = self.frame.frame_bundle.get_local_trivialization_map(p)
        self.coframe_lt_map = self.coframe_bundle.get_local_trivialization_map(p)

        # Get the frame at p
        Ep = self.frame.apply_to_point(p)

        # Apply a local trivialization to get to coordinates
        p, Ep_matrix = self.frame_lt_map(Ep)

        # Get the coframe coordinates
        ep_matrix = jnp.linalg.inv(Ep_matrix)

        # Go back through the coframe local trivialization
        return self.coframe_lt_map.inverse((p, ep_matrix))

    return DualCoframe(self)

  def to_vector_field_list(self) -> List[VectorField]:
    """Turn the frame into a list of vector fields

    Returns:
      List of vector fields
    """
    class FrameElementVectorField(VectorField):
      def __init__(self, i, frame):
        self.i = i
        self.frame = frame
        super().__init__(frame.manifold)

      def apply_to_point(self, p: Point) -> TangentVector:
        vector_list = self.frame(p)
        return vector_list[self.i]

    return [FrameElementVectorField(i, self) for i in range(self.manifold.dimension)]

class StandardCoordinateFrame(Frame):
  """The standard coordinate frame.  The frame at each point
  is a set of tangent vectors that span the tangent space.

  Attributes:
    M: Manifold
  """
  def apply_to_point(self, p: Point) -> TangentBasis:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    lt_map = self.frame_bundle.get_local_trivialization_map(p)
    eye = jnp.eye(self.domain.dimension)
    return lt_map.inverse((p, eye))

################################################################################################################

class Differential(LinearMap[TangentVector,TangentVector]):
  """The differential of a smooth map.

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

    self.TpM = TangentSpace(self.p, M=F.domain)
    self.TpN = TangentSpace(self.q, M=F.image)
    Map.__init__(self, f=self.__call__, domain=self.TpM, image=self.TpN)

  def __call__(self, v: TangentVector) -> TangentVector:
    """Apply the differential to a tangent vector.

    Args:
      v: A tangent vector on M

    Returns:
      dF_p(v)
    """
    assert isinstance(v, TangentVector)

    # Recall that the vector coordinates are in the chart basis.
    # Need to use the coordinate map to get the coordinate change.
    F_hat = self.F.get_coordinate_map(self.p)

    phi = self.domain.manifold.get_chart_for_point(self.p)
    p_hat = phi(self.p)

    test_out = self.F(self.p)
    test_out_coords = F_hat(p_hat)

    # Find the coordinates of the tangent vector on N
    q_hat, z = jax.jvp(F_hat.f, (p_hat,), (v.x,))
    return TangentVector(z, self.TpN)

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
    return jax.jacobian(cm)(p_coords)
