from functools import partial
from typing import Callable, List, Optional, Union, Tuple
from collections import namedtuple
import src.util
from functools import partial, reduce
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
from src.lie_group import *
from src.lie_algebra import *
from src.differential_form import *
from src.instances.manifolds import EuclideanManifold
import src.util as util
import einops
import itertools
import abc
import math

__all__ = ["LieAlgebraValuedAlternatingTensor",
           "LieAlgebraValuedDifferentialForm",
           "maurer_cartan_form",
           "pullback_lie_algebra_form"]

################################################################################################################

class LieAlgebraValuedAlternatingTensor():
  """An alternating tensor that takes vector values. We'll represent this
  using a basis for the vector space and keeping an alternating tensor coefficient
  for each element of the basis.

  Attributes:
    TkTpM: The tensor space of the form
    V: The vector space
  """
  def __init__(self, ws: List[AlternatingTensor], basis: List[LeftInvariantVectorField]):
    """Creates a new vector valued alternating tensor.

    Args:
      xs: A list of coordinates that will be used to create each alternating tensor
      TkTpM: The tensor space
      V: The vector space of the outputs
    """
    self.manifold = ws[0].manifold
    assert all([w.manifold == self.manifold for w in ws])
    self.ws = ws
    self.TkTpM = self.ws[0].TkTpM

    assert all([isinstance(E, LeftInvariantVectorField) for E in basis])
    self.basis_vector_fields = basis

    assert len(self.ws) == len(self.basis_vector_fields)

  def __call__(self, *Xs: List[TangentVector]) -> LeftInvariantVectorField:
    """Apply this tensor to a list of tangent vectors

    Args:
      Xs: A list of tangent vectors

    Returns:
      A vector
    """
    assert all([isinstance(X, TangentVector) for X in Xs])
    assert len(Xs) == self.TkTpM.type.l

    for i, (w, E) in enumerate(zip(self.ws, self.basis_vector_fields)):
      term = w(*Xs)*E
      out = term if i == 0 else out + term

    assert isinstance(out, LeftInvariantVectorField)
    return out

################################################################################################################

class LieAlgebraValuedDifferentialForm(DifferentialForm):
  """The symmetrization of a tensor field.  This will have a non alternating tensor
  field that we will just alternate when calling this tensor.

  TODO: THIS IS A SECTION OF E TENSOR ALTERNATING K FORMS!

  Attributes:
    type: The type of the tensor field
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, ws: List[DifferentialForm], basis: List[LeftInvariantVectorField]):
    """Creates a new vector valued alternating tensor.

    Args:
      xs: A list of coordinates that will be used to create each alternating tensor
      TkTpM: The tensor space
      V: The vector space of the outputs
    """
    self.manifold = ws[0].manifold
    tensor_type = ws[0].type
    assert all([w.manifold == self.manifold for w in ws])
    assert all([w.type == tensor_type for w in ws])

    self.ws = ws

    assert all([isinstance(E, LeftInvariantVectorField) for E in basis])
    self.basis_vector_fields = basis

    self.lieG = self.basis_vector_fields[0].G.lieG

    assert len(self.ws) == len(self.basis_vector_fields)

    super().__init__(tensor_type, self.manifold)

  def apply_to_point(self, p: Point) -> LieAlgebraValuedAlternatingTensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    # Get the values of the forms at p
    wps = [w(p) for w in self.ws]
    return LieAlgebraValuedAlternatingTensor(wps, self.basis_vector_fields)

  def apply_to_co_vector_fields(self, *Xs: List[VectorField]) -> Map:
    """Evaluate the tensor field on vector fields

    Args:
      Xs: A list of vector fields

    Returns:
      A map over the manifold
    """
    def fun(p: Point):
      return self(p)(*[X(p) for X in Xs])
    return Map(fun, domain=self.manifold, image=self.lieG)

################################################################################################################

class MaurerCartanForm(LieAlgebraValuedDifferentialForm):
  """The Maurer Cartan form is a differential form that takes values in the lie algebra
  of a lie group.  It is defined as the differential of the left translation map.

  Attributes:
    G: The lie group that the form is defined on
    lieG: The lie algebra of the lie group
  """
  def __init__(self, G: LieGroup):
    """Create a new maurer cartan form.

    Args:
      G: A lie group
    """
    self.G = G
    self.lieG = G.lieG

    tensor_type = TensorType(0, 1)
    DifferentialForm.__init__(self, tensor_type, self.G)

  def apply_to_point(self, g: Point) -> LieAlgebraValuedAlternatingTensor:
    """
    (w0)_g = d(L_{g^{-1}})_g.

    Args:
      g: Point on the Lie group.

    Returns:
      Tensor at g.
    """
    g_inv = self.G.inverse(g)
    Lg_inv = self.G.left_translation_map(g_inv)
    dLg_inv_g = Differential(Lg_inv, g)

    class MCTensor(LieAlgebraValuedAlternatingTensor):
      def __init__(self, G, dLg_inv_g):
        self.G = G
        self.dLg_inv_g = dLg_inv_g

      def __call__(self, Xg: TangentVector) -> LeftInvariantVectorField:
        assert isinstance(Xg, TangentVector)
        Xe = dLg_inv_g(Xg)
        return LeftInvariantVectorField(Xe, self.G)

    return MCTensor(self.G, dLg_inv_g)

def maurer_cartan_form(G: LieGroup) -> "LieAlgebraValuedDifferentialForm":
  """Return the Maurer cartan form of G

  Args:
    G: A lie group

  Returns:
    The Maurer cartan form of G
  """
  return MaurerCartanForm(G)

################################################################################################################

class PullbackLieAlgebraForm(LieAlgebraValuedDifferentialForm):

  def __init__(self, F: Map, w: LieAlgebraValuedDifferentialForm):
    """Create a new pullback form.

    Args:
      F: A map from a manifold to a manifold
      w: A lie algebra valued differential form
    """
    assert isinstance(F, Map)
    assert isinstance(w, LieAlgebraValuedDifferentialForm)
    assert F.image == w.manifold

    self.F = F
    self.w = w

    tensor_type = w.type
    DifferentialForm.__init__(self, tensor_type, self.F.domain)

  def apply_to_point(self, p: Point) -> LieAlgebraValuedAlternatingTensor:
    """Apply the form to a point on the manifold

    Args:
      A point on the manifold

    Returns:
      The form at p
    """
    class PullbackTensor(LieAlgebraValuedAlternatingTensor):
      def __init__(self, F, w, p):
        self.F = F
        self.w = w
        self.p = p

      def __call__(self, Xp: TangentVector) -> LeftInvariantVectorField:
        assert isinstance(Xp, TangentVector)
        Xq = self.F.get_differential(self.p)(Xp)
        q = self.F(self.p)
        return self.w(q)(Xq)

    return PullbackTensor(self.F, self.w, p)

def pullback_lie_algebra_form(F: Map, w: LieAlgebraValuedDifferentialForm) -> "LieAlgebraValuedDifferentialForm":
  """Return the pullback of the lie algebra valued form w by the map F

  Args:
    F: A map from a manifold to a manifold
    w: A lie algebra valued differential form

  Returns:
    The pullback of w by F
  """
  return PullbackLieAlgebraForm(F, w)