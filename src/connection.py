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
           "maurer_cartan_form"]

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

  def __init__(self, G: LieGroup):
    self.G = G
    self.lieG = G.lieG

    # Get a basis for lieG
    basis_vector_fields = self.lieG.get_basis()

    # Get the dual basis
    dual_basis_covector_fields = self.lieG.get_dual_basis()
    dual_basis = [as_tensor_field(w) for w in dual_basis_covector_fields]
    super().__init__(dual_basis, basis_vector_fields)

  def apply_to_co_vector_fields(self, *Xs: LeftInvariantVectorField) -> Map[Point,LeftInvariantVectorField]:
    """Evaluate the tensor field on vector fields

    Args:
      Xs: A list of vector fields

    Returns:
      A map over the manifold
    """
    def fun(p: Point):
      return self(p)(*[X(p) for X in Xs])
    return Map(fun, domain=self.manifold, image=self.lieG)

def maurer_cartan_form(G: LieGroup) -> "LieAlgebraValuedDifferentialForm":
  return MaurerCartanForm(G)

################################################################################################################
