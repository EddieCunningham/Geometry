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
           "pushforward",
           "lie_bracket"]

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

  def __call__(self, q: Point) -> TangentVector:
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
  return PushforwardVectorField(F, X)

################################################################################################################

class LieBracketVectorField(VectorField):
  """The vector field that is the Lie bracket of X and Y

  Args:
    X: Vector field 1
    Y: Vector field 2

  Returns:
    [X, Y]
  """
  def __init__(self, X: VectorField, Y: VectorField):
    assert X.manifold == Y.manifold
    self.X = X
    self.Y = Y
    super().__init__(M=X.manifold)

  def __call__(self, p: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    # The jvps should be on coordinates
    Xp_nonlocal = self.X(p)
    p_hat = Xp_nonlocal.p_hat

    # Need a differentiable functions to get the coordinates
    def get_x_coordinates(p_hat):
      p = Xp_nonlocal.phi_inv(p_hat)
      Xp = self.X(p)
      return Xp.x

    def get_y_coordinates(p_hat):
      p = Xp_nonlocal.phi_inv(p_hat)
      Yp = self.Y(p)
      # Get the coordinates in terms of the x basis
      return Yp(Xp_nonlocal.phi)

    # First term X^i dY^j/dx^i
    y_coords, X_dYdx = jax.jvp(get_y_coordinates, (p_hat,), (Xp_nonlocal.x,))

    # Get the second term Y^i dX^j/dx^i
    _, Y_dXdx = jax.jvp(get_x_coordinates, (p_hat,), (y_coords,))

    # Compute the Lie bracket coordinates
    new_coords = X_dYdx - Y_dXdx

    return TangentVector(new_coords, Xp_nonlocal.TpM)

def lie_bracket(X: VectorField, Y: VectorField) -> LieBracketVectorField:
  """Compute the Lie bracket [X, Y]

  Args:
    X: Vector field 1
    Y: Vector field 2

  Returns:
    [X, Y]
  """
  return LieBracketVectorField(X, Y)

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
