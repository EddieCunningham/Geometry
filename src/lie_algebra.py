from functools import partial
from typing import Callable, List, Optional, Union
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
from src.bundle import *
from src.flow import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util

__all__ = ["LieAlgebra",
           "LeftInvariantVectorField",
           "induced_lie_algebra_homomorphism",
           "InducedLieAlgebraHomomorphismLIVF",
           "SpaceOfVectorFields",
           "SpaceOfLieAlgebraLinearMaps",
           "SpaceOfMatrices"]

class LeftInvariantVectorField(VectorField):
  """The left invariant vector field corresponding to an
  element of the Lie algebra

  Attributes:
    v: The vector in TeG
    G: The Lie group
  """
  def __init__(self, v: TangentVector, G: LieGroup):
    """Create the object

    Args:
      dim: Dimension
    """
    assert isinstance(v, TangentVector)
    assert isinstance(G, LieGroup)
    self.v = v
    self.G = G
    self.manifold = self.G
    super().__init__(self.manifold)

  def apply_to_point(self, g: Point) -> TangentVector:
    """Evaluate the vector field at a point.
    Basially pushforward of X_e through the map L_g: h |--> g*h

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at g.
    """
    dLg_e = self.G.left_translation_map(g).get_differential(self.G.e)
    return dLg_e(self.v)

  def __add__(self, Y: "LeftInvariantVectorField") -> "LeftInvariantVectorField":
    """Add two sections together

    Args:
      Y: Another LeftInvariantVectorField

    Returns:
      X + Y
    """
    # Ensure that X and Y are compatable
    assert self.manifold == Y.manifold
    # assert self.total_space == Y.total_space

    class SectionSum(LeftInvariantVectorField):
      def __init__(self, X, Y, pi):
        self.X = X
        self.Y = Y
        self.G = X.G
        Section.__init__(self, pi)

      def apply_to_point(self, p: Input) -> Output:
        return self.X(p) + self.Y(p)

    return SectionSum(self, Y, self.pi)

  def __rmul__(self, f: Union[Map,float]) -> "LeftInvariantVectorField":
    """Multiply a LeftInvariantVectorField with a scalar or function. fX

    Args:
      f: Another map or a scalar

    Returns:
      fX
    """
    is_map = isinstance(f, Map)
    is_scalar = f in Reals(dimension=1)

    assert is_map or is_scalar

    class SectionRHSProduct(LeftInvariantVectorField):
      def __init__(self, X, pi):
        self.X = X
        self.G = X.G
        self.lhs = f
        self.is_float = f in Reals(dimension=1)
        Section.__init__(self, pi)

      def apply_to_point(self, p: Input) -> Output:
        fp = self.lhs if self.is_float else self.lhs(p)
        Xp = self.X(p)
        return fp*Xp

    return SectionRHSProduct(self, self.pi)

################################################################################################################

def induced_lie_algebra_homomorphism(F, X):
  return InducedLieAlgebraHomomorphismLIVF(F, X)

class InducedLieAlgebraHomomorphismLIVF(LeftInvariantVectorField):
  """The left invariant vector field corresponding to an
  element of the Lie algebra

  Attributes:
    v: The vector in TeG
    G: The Lie group
  """
  def __init__(self, F: Map, X: LeftInvariantVectorField):
    """Create the object

    Args:
      dim: Dimension
    """
    assert isinstance(F.domain, LieGroup) and isinstance(F.image, LieGroup)
    self.F = F

    # Get the push forward the tangent vector at the identity
    dFe = F.get_differential(X.G.e)
    Xe = X(X.G.e)
    Ye = dFe(Xe)

    super().__init__(Ye, self.F.image)

  def apply_to_point(self, g: Point) -> TangentVector:
    """Evaluate the vector field at a point.
    Basially pushforward of X_e through the map L_g: h |--> g*h

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at g.
    """
    return super().apply_to_point(g)

################################################################################################################

class LieAlgebra(Manifold, abc.ABC):
  """The Lie algebra of a Lie group is the algebra of all smooth left-invariant vector
  fields on a Lie group G.  It is isomorphic to the tangent space of G at the identity.

  Attributes:
    G: The Lie group
  """
  Element = LeftInvariantVectorField

  def __init__(self, G: LieGroup):
    """Create a Lie algebra object

    Args:
      G: The corresponding Lie group
    """
    self.G = G
    self.TeG = TangentSpace(self.G.e, self.G)
    super().__init__(dimension=self.G.dimension)

  def contains(self, p: Point) -> bool:
    """Checks to see if p exists in the manifold.  This is the case if
       the point is in the domain of any chart

    Args:
      p: Test point.

    Returns:
      True if p is in the manifold, False otherwise.
    """
    type_check = isinstance(p, LeftInvariantVectorField)
    manifold_check = self.G == p.G
    return type_check and manifold_check

  def get_atlas(self) -> "Atlas":
    """Construct the atlas for a manifold

    Attributes:
      atlas: Atlas providing coordinate representation.
    """
    def chart_fun(v, inverse=False):
      if inverse == False:
        assert isinstance(v, LeftInvariantVectorField)
        # Evaluate at the identity
        ve = v(self.G.e)
        return ve.x
      else:
        # Create tangent vector at identity
        ve = TangentVector(v, self.TeG)
        return self.get_left_invariant_vector_field(ve)

    self.chart = Chart(chart_fun, domain=self, image=Reals(dimension=self.dimension))
    return Atlas([self.chart])

  @abc.abstractmethod
  def bracket(self, X: Point, Y: Point) -> Point:
    """A real vector space endowed with a bracket

    Args:
      X: Element of lie algebra
      Y: Element of lie algebra

    Returns:
      Z: New element of lie algebra
    """
    pass

  def get_left_invariant_vector_field(self, v: TangentVector) -> LeftInvariantVectorField:
    """Given v in TeG, return the vector field V where V_g = d(L_g)_e(v)

    Args:
      v: A tangent vector at the identity

    Returns:
      A vector field where at point g, V_g = d(L_g)_e(v)
    """
    assert isinstance(v, TangentVector)
    assert v.TpM == TangentSpace(self.G.e, self.G)
    return LeftInvariantVectorField(v, self.G)

  def get_one_parameter_subgroup(self, X: LeftInvariantVectorField) -> IntegralCurve:
    """One parameter subgroup generated by X.  These are maximal integral curves
    of left invariant vector fields starting at the identity.

    Args:
      X: A left invariant vector field

    Returns:
      The integral curve generated by X
    """
    assert isinstance(X, LeftInvariantVectorField)
    return IntegralCurve(self.G.e, X)

  def get_exponential_map(self) -> Map[LeftInvariantVectorField,Point]:
    """exp: Lie(G) -> G is defined by exp(X) = one_parameter_subgroup(1.0).
    This turns a left invariant vector field into an element of the Lie group.

    Returns:
      The exponential map
    """
    def f(X: LeftInvariantVectorField):
      ops = self.get_one_parameter_subgroup(X)
      return ops(1.0)
    return Map(f, domain=self, image=self.G)

  def get_adjoint_representation(self) -> Map[Point,LinearMap]:
    """Get the adjoint representation of this Lie algebra.  This turns a
    left invariant vector field into a matrix lie algebra.

    Returns:
      Adjoint representation
    """
    def _ad(X: LeftInvariantVectorField):
      return LinearMap(lambda Y: self.bracket(X, Y), domain=self, image=self)

    ad = Map(_ad, domain=self, image=SpaceOfLieAlgebraLinearMaps(self.G))
    return ad

################################################################################################################

class SpaceOfVectorFields(LieAlgebra):

  def bracket(self, X: VectorField, Y: VectorField) -> VectorField:
    """This is just the lie bracket

    Args:
      X: Element of lie algebra
      Y: Element of lie algebra

    Returns:
      Z: New element of lie algebra
    """
    return lie_bracket(X, Y)

  def contains(self, p: VectorField):
    return isinstance(p, VectorField)

################################################################################################################

class SpaceOfMatrices(LieAlgebra):

  def __init__(self, G: GLRn):
    """Create the space of nxn matrices.

    Args:
      dim: Dimension
    """
    assert isinstance(G, GLRn)
    super().__init__(G)

  def bracket(self, A: LeftInvariantVectorField, B: LeftInvariantVectorField) -> LeftInvariantVectorField:
    """Commutator bracket [A,B] = AB - BA

    Attributes:
      X: Element of lie algebra
      Y: Element of lie algebra

    Returns:
      Z: New element of lie algebra
    """
    dim = self.G.N
    assert isinstance(A, LeftInvariantVectorField)
    assert isinstance(B, LeftInvariantVectorField)
    assert A.G == B.G
    G = A.G
    lieG = G.get_lie_algebra()

    # Apply the lie bracket at the identity
    Ae = A(A.G.e)
    Be = B(B.G.e)
    a = Ae.x.reshape((dim, dim))
    b = Be.x.reshape((dim, dim))
    c = a@b - b@a

    Ce = TangentVector(c.ravel(), lieG.TeG)
    C = lieG.get_left_invariant_vector_field(Ce)
    return C

  def contains(self, p: LeftInvariantVectorField) -> bool:
    type_check = isinstance(p, LeftInvariantVectorField)
    group_check = p.G == self.G
    return type_check and group_check

################################################################################################################

class SpaceOfLieAlgebraLinearMaps(Set):

  Element = LinearMap

  def __init__(self, G: LieGroup):
    """Create the space of nxn matrices.

    Args:
      dim: Dimension
    """
    self.G = G
    super().__init__()

  def contains(self, p: LinearMap) -> bool:
    return True
    type_check = isinstance(p, LinearMap)
    domain_check = p.domain == self.G
    image_check = p.image == self.G
    out = type_check and domain_check and image_check
    if out == False:
      import pdb; pdb.set_trace()
    return out