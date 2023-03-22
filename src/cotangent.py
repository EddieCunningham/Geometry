from functools import partial
from typing import Callable, List, Optional
import src.util
from functools import partial
import jax
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.tangent import *
from src.instances.manifolds import Vector, VectorSpace
import src.util as util

__all__ = ["CotangentVector",
           "CotangentSpace",
           "CotangentBasis",
           "Pullback"]

class CotangentVector(LinearMap[TangentVector,Coordinate]):
  """A covector on a Manifold

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    coTpM: The tangent space that the vector lives on.
  """
  def __init__(self, x: Coordinate, coTpM: "CotangentSpace"):
    """Creates a new covector.

    Args:
      x: The coordinates of the vector in the basis induced by a chart.
      coTpM: The tangent space that the vector lives on.
    """
    self.x = x
    self.coTpM = coTpM
    assert x.shape[0] == self.coTpM.dimension
    self.phi = self.coTpM.phi

  @property
  def phi_inv(self):
    return self.phi.get_inverse()

  @property
  def p(self):
    """The point that the covector is defined at.
    """
    return self.coTpM.p

  @property
  def p_hat(self):
    return self.phi(self.p)

  def __call__(self, v: TangentVector) -> Coordinate:
    """Apply the covector to a tangent vector v

    Args:
      v: Input tangent vector

    Returns:
      w(v)
    """
    assert isinstance(v, TangentVector)
    assert v.TpM.get_dual_space() == self.coTpM

    # Evaluate the vector components in this basis
    v_coords = v(self.phi)
    return jnp.vdot(self.x, v_coords)

  def __add__(self, Y: "CotangentVector") -> "CotangentVector":
    """Add two tangent vectors together

    Args:
      Y: Another tangent vector

    Returns:
      Xp + Yp
    """
    # Can only add if they are a part of the same tangent space
    assert self.coTpM == Y.coTpM

    # Need to coordinates of Y in the basis we're using for X
    y_coords = Y(self.phi)

    # Add the coordinates together
    return CotangentVector(self.x + y_coords, self.coTpM)

  def __radd__(self, Y: "CotangentVector") -> "CotangentVector":
    """Add Y from the right

    Args:
      Y: Another cotangent vector

    Returns:
      X + Y
    """
    return self + Y

  def __rmul__(self, a: float) -> "CotangentVector":
    """Cotangent vectors support scalar multiplication

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    if util.GLOBAL_CHECK:
      (a in Reals(dimension=1))
    return CotangentVector(self.x*a, self.coTpM)

  def __sub__(self, Y: "CotangentVector") -> "CotangentVector":
    """Subtract Y from this vector

    Args:
      Y: Another cotangent vector

    Returns:
      X - Y
    """
    return self + -1.0*Y

  def __neg__(self):
    """Negate this vector

    Returns:
      Negative of this vector
    """
    return -1.0*self

class CotangentSpace(VectorSpace):
  """Cotangent space of a manifold.  Is defined at a point, T_p^*M

  Attributes:
    p: The point where the space lives.
    M: The manifold.
  """
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

    # We also need a chart for the tangent space.  Because we're
    # already using coordinates to represent the tangent vectors,
    # we don't need to do anything special.
    def chart_fun(v, inverse=False):
      if inverse == False:
        return v.x
      else:
        return TangentVector(v, self)

    self.dimension = self.manifold.dimension
    self.chart = Chart(chart_fun, domain=self, image=Reals(dimension=self.dimension))
    super().__init__(dimension=self.manifold.dimension, chart=self.chart)

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

  def get_basis(self) -> List[CotangentVector]:
    """Get a basis of vectors for the cotangent space

    Returns:
      A list of cotangent vectors that form a basis for the cotangent space
    """
    eye = jnp.eye(self.dimension)
    basis = []
    for i in range(self.dimension):
      v = CotangentVector(eye[i], coTpM=self)
      basis.append(v)
    return basis

  def get_dual_space(self) -> TangentSpace:
    """Get the dual space corresponding to the cotangent space

    Returns:
      Tangent space
    """
    from src.tangent import TangentSpace
    return TangentSpace(self.p, self.manifold)

################################################################################################################

class CotangentBasis(InvertibleMatrix):
  """A list of cotangent vectors that forms a basis for the cotangent space.

  Attributes:
    basis: A list of cotangent vectors
    coTpM: The cotangent space that the vector lives on.
  """
  def __init__(self, Xs: List[CotangentVector], coTpM: CotangentSpace):
    self.basis = Xs
    self.coTpM = coTpM
    assert len(self.basis) == self.coTpM.dimension

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
    frame_bundle = FrameBundle(TpM)
    lt_map = frame_bundle.get_local_trivialization_map(tangent_basis)
    p, matrix = lt_map(tangent_basis)

    # Invert
    inv_matrix = jnp.linalg.inv(matrix)

    # Now use a local trivialization of the coframe bundle to go to
    # the cotangent basis
    coframe_bundle = CoframeBundle(TpM.manifold)
    co_lt_map = coframe_bundle.get_local_trivialization_map(p) # TODO: Whats the right thing to pass in here?
    return co_lt_map.inverse((p, inv_matrix))

  def __add__(self, Y: "CotangentBasis") -> "CotangentBasis":
    """Add two cotangent space bases.

    Args:
      Y: Another cotangent basis

    Returns:
      Xp + Yp
    """
    # Can only add if they are a part of the same cotangent space
    assert self.coTpM == Y.coTpM

    new_basis = [u + v for u, v in zip(self.basis, Y.basis)]
    return CotangentBasis(*new_basis, self.coTpM)

  def __radd__(self, Y: "CotangentBasis") -> "CotangentBasis":
    """Add Y from the right

    Args:
      Y: Another cotangent basis

    Returns:
      X + Y
    """
    return self + Y

  def __rmul__(self, a: float) -> "CotangentBasis":
    """cotangent basis support scalar multiplication

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    if util.GLOBAL_CHECK:
      (a in Reals(dimension=1))

    new_basis = [a*v for v in self.basis]
    return CotangentBasis(new_basis, self.coTpM)

  def __sub__(self, Y: "CotangentBasis") -> "CotangentBasis":
    """Subtract Y from this basis

    Args:
      Y: Another cotangent basis

    Returns:
      X - Y
    """
    return self + -1.0*Y

  def __neg__(self):
    """Negate this basis

    Returns:
      Negative of this basis
    """
    return -1.0*self

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
      dF_p(v)
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
