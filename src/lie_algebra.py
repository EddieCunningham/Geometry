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
from src.cotangent import *
from src.vector import *
from src.vector_field import *
from src.section import *
from src.bundle import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util

__all__ = ["LieAlgebra",
           "LeftInvariantVectorField",
           "LeftInvariantCovectorField",
           "induced_lie_algebra_homomorphism",
           "InducedLieAlgebraHomomorphismLIVF",
           "SpaceOfVectorFields",
           "SpaceOfLieAlgebraLinearMaps",
           "SpaceOfMatrices"]

class LeftInvariantCovectorField(CovectorField):
  """The left invariant vector field corresponding to an
  element of the Lie algebra

  Attributes:
    v: The vector in TeG
    G: The Lie group
  """
  def __init__(self, w: CotangentVector, G: LieGroup):
    """Create the object

    Args:
      dim: Dimension
    """
    assert isinstance(w, CotangentVector)
    assert isinstance(G, LieGroup)
    self.w = w
    self.G = G
    self.manifold = self.G
    super().__init__(self.manifold)

  def apply_to_point(self, g: Point) -> CotangentVector:
    """Evaluate the vector field at a point.
    Pull w back from TeG to TgG

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at g.
    """
    if g is self.G.e:
      return self.w

    assert g in self.G
    g_inv = self.G.inverse(g)
    dLg_e = self.G.left_translation_map(g_inv).get_pullback(g)
    return dLg_e(self.w)

  def __add__(self, Y: "LeftInvariantCovectorField") -> "LeftInvariantCovectorField":
    """Add two sections together

    Args:
      Y: Another LeftInvariantCovectorField

    Returns:
      X + Y
    """
    # Ensure that X and Y are compatable
    assert self.manifold == Y.manifold
    # assert self.total_space == Y.total_space

    class LICFSum(LeftInvariantCovectorField):
      def __init__(self, X, Y):
        assert isinstance(X, LeftInvariantCovectorField)
        assert isinstance(Y, LeftInvariantCovectorField)
        assert X.G == Y.G
        self.X = X
        self.Y = Y
        e = X.G.e
        w = X(e) + Y(e)
        super().__init__(w, X.G)

    return LICFSum(self, Y)

  def __rmul__(self, f: Union[Map,float]) -> "LeftInvariantCovectorField":
    """Multiply a LeftInvariantCovectorField with a scalar or function. fX

    Args:
      f: Another map or a scalar

    Returns:
      fX
    """
    is_map = isinstance(f, Map)
    is_scalar = f in EuclideanSpace(dimension=1)

    if is_scalar:
      class LIVFScalarProduct(LeftInvariantCovectorField):
        def __init__(self, a, X):
          assert isinstance(X, LeftInvariantCovectorField)
          self.a = a
          self.X = X
          G = self.X.G
          w = a*self.X(G.e)
          super().__init__(w, G)

      return LIVFScalarProduct(f, self)

    else:
      # No longer left invariant
      class LIVFMapProduct(CovectorField):
        def __init__(self, f, X):
          assert isinstance(X, LeftInvariantCovectorField)
          assert isinstance(f, Map)
          self.X = X
          self.G = X.G
          self.f = f
          super().__init__(self.G)

        def apply_to_point(self, p: Input) -> Output:
          fp = self.f(p)
          Xp = self.X(p)
          return fp*Xp

      return LIVFMapProduct(f, self)

################################################################################################################

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
    if g is self.G.e:
      return self.v

    assert g in self.G
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

    class LIVFSum(LeftInvariantVectorField):
      def __init__(self, X, Y):
        assert isinstance(X, LeftInvariantVectorField)
        assert isinstance(Y, LeftInvariantVectorField)
        assert X.G == Y.G
        self.X = X
        self.Y = Y
        e = X.G.e
        v = X(e) + Y(e)
        super().__init__(v, X.G)

    return LIVFSum(self, Y)

  def __rmul__(self, f: Union[Map,float]) -> "LeftInvariantVectorField":
    """Multiply a LeftInvariantVectorField with a scalar or function. fX

    Args:
      f: Another map or a scalar

    Returns:
      fX
    """
    is_map = isinstance(f, Map)
    is_scalar = f in EuclideanSpace(dimension=1)

    if is_scalar:
      class LIVFScalarProduct(LeftInvariantVectorField):
        def __init__(self, a, X):
          assert isinstance(X, LeftInvariantVectorField)
          self.a = a
          self.X = X
          G = self.X.G
          v = a*self.X(G.e)
          super().__init__(v, G)

      return LIVFScalarProduct(f, self)

    else:
      # No longer left invariant
      class LIVFMapProduct(VectorField):
        def __init__(self, f, X):
          assert isinstance(X, LeftInvariantVectorField)
          assert isinstance(f, Map)
          self.X = X
          self.G = X.G
          self.f = f
          super().__init__(self.G)

        def apply_to_point(self, p: Input) -> Output:
          fp = self.f(p)
          Xp = self.X(p)
          return fp*Xp

      return LIVFMapProduct(f, self)

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
    self.G = X.G
    self.e = self.G.e

    # Get the push forward the tangent vector at the identity
    dFe = F.get_differential(self.e)
    self.Xe = X(self.e)
    self.Ye = dFe(self.Xe)

    super().__init__(self.Ye, self.F.image)

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

  def get_chart_for_point(self, p: Point) -> "Chart":
    """Get a chart to use at point p

    Args:
      The input point

    Returns:
      The chart that contains p in its domain
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

    return Chart(chart_fun, domain=self, image=EuclideanSpace(dimension=self.dimension))

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

  def get_left_invariant_covector_field(self, w: CotangentVector) -> LeftInvariantCovectorField:
    """Given w in coTeG, return the vector field W where W_g = d(L_g)_e^*(w)

    Args:
      w: A cotangent vector at the identity

    Returns:
      A covector field where at point g, W_g = d(L_g)_e^*(w)
    """
    assert isinstance(w, CotangentVector)
    assert w.coTpM == CotangentSpace(self.G.e, self.G)
    return LeftInvariantCovectorField(w, self.G)

  def get_one_parameter_subgroup(self, X: LeftInvariantVectorField) -> "IntegralCurve":
    """One parameter subgroup generated by X.  These are maximal integral curves
    of left invariant vector fields starting at the identity.

    Args:
      X: A left invariant vector field

    Returns:
      The integral curve generated by X
    """
    from src.flow import IntegralCurve
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

  def get_TeG_basis(self) -> List[TangentVector]:
    """Get a basis for the tangent space at the identity.  This will be used
    to get a basis for the lie algebra

    Returns:
      A list of tangent vectors
    """
    return self.TeG.get_basis()

  def get_basis(self) -> List[VectorField]:
    """Get a basis for the lie algebra.

    Returns:
      A list of vector fields
    """
    return [self.get_left_invariant_vector_field(v) for v in self.get_TeG_basis()]

  def get_TeG_dual_basis(self) -> List[CotangentVector]:
    """Get a dual basis for the tangent space at the identity.

    Returns:
      A list of cotangent vectors
    """
    return self.TeG.get_dual_basis()

  def get_dual_basis(self) -> List[CovectorField]:
    """Get a dual basis for the lie algebra.

    Returns:
      A list of covector fields
    """
    return [self.get_left_invariant_covector_field(w) for w in self.get_TeG_dual_basis()]

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
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    return isinstance(p, VectorField)

################################################################################################################

class SpaceOfMatrices(LieAlgebra):

  Element = Matrix

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
    assert isinstance(p, LeftInvariantVectorField)
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

  def get_TeG_basis(self) -> List[TangentVector]:
    """Get a basis for the tangent space at the identity.  This will be used
    to get a basis for the lie algebra

    Returns:
      A list of tangent vectors
    """
    assert 0, "Not implemented"
