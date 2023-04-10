from functools import partial
from typing import Callable, List, Optional, Union
import src.util
from functools import partial
from copy import deepcopy
import jax
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
from src.tangent import *
from src.section import *
from src.bundle import *
import src.util as util

__all__ = ["TimeDependentVectorField",
           "PushforwardVectorField",
           "pushforward"]

################################################################################################################

class PushforwardVectorField(VectorField):
  """The pushforward of a vector field X through a diffeomorphism F

  Attributes:
    F: Diffeomorphism
    X: Vector field
  """
  def __init__(self, F: Diffeomorphism, X: VectorField):
    """Create a new pushforward vector field object

    Args:
      F: Diffeomorphism
      X: Vector field
    """
    assert X.manifold == F.domain
    self.F = F
    self.X = X
    super().__init__(M=F.image)

  def apply_to_point(self, q: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      q: Point on the manifold.

    Returns:
      Tangent vector at q.
    """
    p = self.F.inverse(q)
    Xp = self.X(p)
    dFp = self.F.get_differential(p)
    return dFp(Xp)

def pushforward(F: Map, X: VectorField) -> PushforwardVectorField:
  """The pushforward of X by F.  F_*(X)

  Args:
    F: A map
    X: A vector field on defined on the image of F

  Returns:
    F_*(X)
  """
  from src.lie_algebra import LeftInvariantVectorField
  if isinstance(X, LeftInvariantVectorField):
    from src.lie_group import LieGroup
    assert isinstance(F.domain, LieGroup) and isinstance(F.image, LieGroup)
    # Just requires us to push forward the tangent vector at the identity
    dFe = F.get_differential(X.G.e)
    Yv = dFe(X.v)
    return LeftInvariantVectorField(Yv, F.image)

  assert isinstance(X, VectorField)
  return PushforwardVectorField(F, X)

################################################################################################################

class TimeDependentVectorField():
  pass

class TimeDependentVectorField(VectorField):

  @abc.abstractmethod
  def __call__(self, t: Coordinate, p: Point) -> TangentVector:
    assert 0

  def __add__(self, Y: TimeDependentVectorField) -> TimeDependentVectorField:
    pass

  def __rmul__(self, f: Union[Map,float]) -> TimeDependentVectorField:
    pass

  def __mul__(self, f: Map) -> Map:
    pass
