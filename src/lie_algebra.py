from functools import partial
from typing import Callable, List, Optional
import src.util
from functools import partial
from copy import deepcopy
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
from src.lie_group import *
from src.tangent import *
from src.vector import *
from src.vector_field import *
from src.section import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util

__all__ = ["_LieAlgebraMixin",
           "LieAlgebra",
           "SpaceOfVectorFields",
           "SpaceOfMatrices"]

class LeftInvariantVectorField(VectorField):
  """The left invariant vector field corresponding to an
  element of the Lie algebra

  Attributes:
    v: The vector in the Lie algebra
    G: The Lie group
  """
  def __init__(self, v: TangentVector, G: LieGroup):
    """Create the object

    Args:
      dim: Dimension
    """
    self.v = v
    self.G = G
    self.manifold = self.G
    super().__init__(self.manifold)

  def __call__(self, g: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    dLe = self.G.left_translation_map(g).get_differential(self.G.e)
    return dLe(self.v)

class _LieAlgebraMixin(abc.ABC):
  """A real vector space endowed with a bracket.
  """

  @abc.abstractmethod
  def bracket(self, X: Point, Y: Point) -> Point:
    """A real vector space endowed with a bracket

    Attributes:
      X: Element of lie algebra
      Y: Element of lie algebra

    Returns:
      Z: New element of lie algebra
    """
    pass

  def left_invariant_vector_field(self, v: TangentVector) -> LeftInvariantVectorField:
    """Given v in TeG, return the vector field V where V_g = d(L_g)_e(v)

    Attributes:
      v: A tangent vector at the identity

    Returns:
      A vector field where at point g, V_g = d(L_g)_e(v)
    """
    if util.GLOBAL_CHECK:
      assert isinstance(v, TangentVector)
      assert v.TpM == TangentSpace(self.G.e, self.G)

    return LeftInvariantVectorField(v, self.G)

class LieAlgebra(TangentSpace, _LieAlgebraMixin):
  """The Lie algebra of a Lie group is the tangent space at the identity

  Attributes:
    G: The Lie group
  """
  Element = Vector

  def __init__(self, G: LieGroup):
    """Create a Lie algebra object

    Args:
      G: The corresponding Lie group
    """
    self.G = G
    super().__init__(G.e, G)

################################################################################################################

class SpaceOfVectorFields(EuclideanManifold, _LieAlgebraMixin):

  def bracket(self, X: VectorField, Y: VectorField) -> VectorField:
    """This is just the lie bracket

    Attributes:
      X: Element of lie algebra
      Y: Element of lie algebra

    Returns:
      Z: New element of lie algebra
    """
    return lie_bracket(X, Y)

  def __contains__(self, p: VectorField):
    return isinstance(p, VectorField)

################################################################################################################

class SpaceOfMatrices(LieAlgebra):

  def __init__(self, dim: int):
    """Create the space of nxn matrices.

    Args:
      dim: Dimension
    """
    self.N = dim
    GLn = GeneralLinearGroup(self.N)
    super().__init__(GLn)

  def bracket(self, A: Point, B: Point) -> Point:
    """Commutator bracket [A,B] = AB - BA

    Attributes:
      X: Element of lie algebra
      Y: Element of lie algebra

    Returns:
      Z: New element of lie algebra
    """
    dim = self.G.N

    class Bracket(VectorField):
      def __init__(self):
        self.A = A
        self.B = B
        super().__init__(A.manifold)

      def __call__(self, p: Point) -> TangentVector:
        Ap, Bp = self.A(p), self.B(p)

        # Get the matrix components of each tangent vector
        a = Ap.x.reshape((dim, dim))
        b = Bp.x.reshape((dim, dim))
        c = a@b - b@a
        return TangentVector(c.ravel(), Ap.TpM)

    return Bracket()

  def __contains__(self, p: Point) -> bool:
    shape_condition = p.shape == (self.N, self.N)
    return shape_condition
