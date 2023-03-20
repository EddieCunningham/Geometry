from functools import partial
from typing import Callable, List, Optional, Generic, Tuple
import src.util
from functools import partial
import jax
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.instances.manifolds import Vector, VectorSpace
import src.util as util

__all__ = ["TangentVector",
           "TangentSpace",
           "MatrixRowsToList",
           "MatrixColsToList",
           "TangentBasis",
           "TangentBasisSpace",
           "Differential",
           "GlobalDifferential"]

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
    self.x = x
    self.TpM = TpM
    assert x.shape[0] == self.TpM.dimension
    self.phi = self.TpM.phi

  @property
  def phi_inv(self):
    """Inverse of the chart
    """
    return self.phi.get_inverse()

  @property
  def p(self):
    """The point that the tangent vector is defined at.
    """
    return self.TpM.p

  @property
  def p_hat(self):
    """The cooordinates for the point
    """
    return self.phi(self.p)

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

  def __add__(self, Y: "TangentVector") -> "TangentVector":
    """Add two tangent vectors together

    Args:
      Y: Another tangent vector

    Returns:
      Xp + Yp
    """
    # Can only add if they are a part of the same tangent space
    assert self.TpM == Y.TpM

    # Need to coordinates of Y in the basis we're using for X
    y_coords = Y(self.phi)

    # Add the coordinates together
    return TangentVector(self.x + y_coords, self.TpM)

  def __radd__(self, Y: "TangentVector") -> "TangentVector":
    """Add Y from the right

    Args:
      Y: Another tangent vector

    Returns:
      X + Y
    """
    return self + Y

  def __rmul__(self, a: float) -> "TangentVector":
    """Tangent vectors support scalar multiplication

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    if util.GLOBAL_CHECK:
      (a in Reals(dimension=1))
    return TangentVector(self.x*a, self.TpM)

  def __sub__(self, Y: "TangentVector") -> "TangentVector":
    """Subtract Y from this vector

    Args:
      Y: Another tangent vector

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

  def __eq__(self, other: "TangentSpace") -> bool:
    """Checks to see if 2 tangent spaces are equal.

    Args:
      other: The other tangent space.

    Returns:
      True if they are the same, False otherwise.
    """
    point_same = True
    if util.GLOBAL_CHECK:
      point_same = jnp.allclose(self.p, other.p)
    return point_same and (self.manifold == other.manifold)

  def __contains__(self, v: TangentVector) -> bool:
    """Checks to see if p exists in the manifold.  This is the case if
       the point is in the domain of any chart

    Args:
      p: Test point.

    Returns:
      True if p is in the manifold, False otherwise.
    """
    type_check = isinstance(v, TangentVector)
    val_check = v.TpM == self
    return type_check and val_check

  def get_basis(self) -> List[TangentVector]:
    """Get a basis of vectors for the tangent space

    Returns:
      A list of tangent vectors that form a basis for the tangent space
    """
    eye = jnp.eye(self.dimension)
    basis = []
    for i in range(self.dimension):
      v = TangentVector(eye[i], TpM=self)
      basis.append(v)
    return basis

  def get_dual_space(self) -> "CotangentSpace":
    """Get the dual space corresponding to the tangent space

    Returns:
      Cotangent space
    """
    from src.cotangent import CotangentSpace
    return CotangentSpace(self.p, self.manifold)

################################################################################################################

from src.map import _InvertibleMixin

class MatrixToList(Map[Matrix,List[TangentVector]], _InvertibleMixin):
  """Returns the matrix as a list split along a specified index.

  Attributes:
    TpM: The tangent space that this is a basis of.
  """
  def __init__(self, TpM: TangentSpace, *, index: int):
    """Creates a new MatrixToList

    Args:
      TpM: The tangent space that this is a basis of.
      index: The index to split on
    """
    self.TpM = TpM
    self.index = index

    def f(v, inverse=False):
      if inverse == False:
        coords = jnp.stack([_v.x for _v in v])
        return v.x
      else:
        _vx = jnp.split(v, self.TpM.dimension, axis=self.index)
        return [TangentVector(vx, self.TpM) for vx in _vx]

    domain = GeneralLinearGroup(dim=self.TpM.dimension)
    image = TangentBasis(self.TpM)
    super().__init__(f, domain=domain, image=image)

class MatrixRowsToList(Map[Matrix,List[TangentVector]], _InvertibleMixin):
  """Returns the rows of a matrix as a list.

  Attributes:
    TpM: The tangent space that this is a basis of.
  """
  def __init__(self, TpM: TangentSpace):
    """Creates a new MatrixRowsToList

    Args:
      TpM: The tangent space that this is a basis of.
    """
    super().__init__(TpM, index=0)

class MatrixColsToList(Map[Matrix,List[TangentVector]], _InvertibleMixin):
  """Returns the columns of a matrix as a list.

  Attributes:
    TpM: The tangent space that this is a basis of.
  """
  def __init__(self, TpM: TangentSpace):
    """Creates a new MatrixColsToList

    Args:
      TpM: The tangent space that this is a basis of.
    """
    super().__init__(TpM, index=1)

################################################################################################################

class TangentBasis(InvertibleMatrix):
  """A list of tangent vectors that forms a basis for the tangent space.

  Attributes:
    basis: A list of tangent vectors
    TpM: The tangent space that the vector lives on.
  """
  def __init__(self, *Xs: List[TangentVector], TpM: TangentSpace):
    self.basis = Xs
    self.TpM = TpM
    assert len(self.basis) == self.TpM.dimension

  def __add__(self, Y: "TangentBasis") -> "TangentBasis":
    """Add two tangent space bases.

    Args:
      Y: Another tangent basis

    Returns:
      Xp + Yp
    """
    # Can only add if they are a part of the same tangent space
    assert self.TpM == Y.TpM

    new_basis = [u + v for u, v in zip(self.basis, Y.basis)]
    return TangentBasis(*new_basis, self.TpM)

  def __radd__(self, Y: "TangentBasis") -> "TangentBasis":
    """Add Y from the right

    Args:
      Y: Another tangent basis

    Returns:
      X + Y
    """
    return self + Y

  def __rmul__(self, a: float) -> "TangentBasis":
    """Tangent basis support scalar multiplication

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    if util.GLOBAL_CHECK:
      (a in Reals(dimension=1))

    new_basis = [a*v for v in self.basis]
    return TangentBasis(new_basis, self.TpM)

  def __sub__(self, Y: "TangentBasis") -> "TangentBasis":
    """Subtract Y from this basis

    Args:
      Y: Another tangent basis

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

  def get_dual_basis(self) -> "DualBasis":
    """Get the corresponding dual basis.

    Returns:
      Corresponding cotangent basis
    """
    # Rows of the inverse matrix of coordinates
    pass

class TangentBasisSpace(VectorSpace):
  """The space that a list of tangent vectors that forms a basis lives in

  Attributes:
    p: The point where the space lives.
    M: The manifold.
  """
  Element = List[TangentVector]

  def __init__(self, TpM: TangentSpace):
    """Creates a new tangent space.

    Args:
      p: The point where the space lives.
      M: The manifold.
    """
    self.TpM = TpM
    self.p = self.TpM.p
    self.manifold = self.TpM.manifold

    # Keep track of the chart function for the manifold
    self.phi = self.manifold.get_chart_for_point(self.p)

    # We also need a chart for the tangent space.  Because we're
    # already using coordinates to represent the tangent vectors,
    # we don't need to do anything special.
    def chart_fun(v, inverse=False):
      if inverse == False:
        coords = jnp.stack([_v.x for _v in v])
        return v.x
      else:
        _vx = jnp.split(v, self.manifold.dimension, axis=0)
        return [TangentVector(vx, self.TpM) for vx in _vx]

    self.dimension = self.TpM.dimension**2
    self.chart = Chart(chart_fun, domain=self, image=Reals(dimension=dim))
    super().__init__(dimension=self.manifold.dimension, chart=self.chart)

  def __eq__(self, other: "TangentBasis") -> bool:
    """Checks to see if 2 tangent spaces are equal.

    Args:
      other: The other tangent space.

    Returns:
      True if they are the same, False otherwise.
    """
    point_same = True
    if util.GLOBAL_CHECK:
      point_same = jnp.allclose(self.p, other.p)
    return point_same and (self.manifold == other.manifold)

################################################################################################################

class Differential(LinearMap):
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
    TpM = TangentSpace(p, M=F.domain)
    TpN = TangentSpace(F(p), M=F.image)
    super().__init__(f=F, domain=TpM, image=TpN)

    self.q = self.F(self.p)

  @property
  def F(self):
    return self.f

  def __call__(self, v: TangentVector) -> TangentVector:
    """Apply the differential to a tangent vector.

    Args:
      v: A tangent vector on M

    Returns:
      dF_p(v)
    """
    assert isinstance(v, TangentVector)

    # Needs to be at same point
    if util.GLOBAL_CHECK:
      assert jnp.allclose(self.p, v.p)

    # Recall that the vector coordinates are in the chart basis.
    # Need to use the coordinate map to get the coordinate change.
    F_hat = self.F.get_coordinate_map(self.p)

    phi = self.domain.manifold.get_chart_for_point(self.p)
    p_hat = phi(self.p)

    # Find the coordinates of the tangent vector on N
    q_hat, z = jax.jvp(F_hat.f, (p_hat,), (v.x,))

    # psi = self.image.manifold.get_chart_for_point(self.q)
    # q_hat_comp = psi(self.q)
    # assert jnp.allclose(q_hat, q_hat_comp)

    TpN = TangentSpace(self.q, self.F.image)
    return TangentVector(z, TpN)

  def get_dual_map(self) -> LinearMap:
    """Get the dual map of the differential

    Args:

    Returns:
      dF_p^*
    """
    dFp = self
    class Transpose(LinearMap):

      def __init__(self):
        self.dFp = dFp
        domain = CotangentSpace(self.dFp.q, self.dFp.image)
        image = CotangentSpace(self.dFp.p, self.dFp.domain)
        super().__init__(f=None, domain=TpM, image=TpN)

      def __call__(self, w: Covector) -> Covector:
        assert 0

    return Transpose()

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

class GlobalDifferential(LinearMap):
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

  def differential(self, p: Point) -> Differential:
    return Differential(self.F, p)
