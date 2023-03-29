from functools import partial
from typing import Callable, List, Optional, Union, Tuple
from collections import namedtuple
import src.util
from functools import partial
import jax
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.tangent import *
from src.cotangent import *
from src.vector import *
from src.section import *
from src.tensor import *
import src.util as util
import einops
import itertools

__all__ = ["LieBracketVectorField",
           "lie_bracket",
           "LieDerivativeTensorField",
           "lie_derivative"]

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

  def apply_to_point(self, p: Point) -> TangentVector:
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

class LieDerivativeTensorField(TensorField):
  """This tensor represents the Lie derivative of a covariant tensor

  Attributes:
    V: The vector field that we're flowing on
    A: The covariant tensor
  """
  def __init__(self, V: VectorField, A: TensorField):
    """Creates a new LieDerivativeTensorField object.

    Args:
      V: Vector field that we're flowing on
      A: The thing to covariant tensor field
    """
    assert A.type.k == 0
    self.V = V
    self.A = A
    super().__init__(self.A.type, self.A.manifold)

  def apply_to_co_vector_fields(self, *Xs: List[VectorField]) -> Map:
    """Evaluate the tensor field on vector fields.  Use corolarry 12.33

    Args:
      Xs: A list of vector fields

    Returns:
      A map over the manifold
    """
    out = self.V(self.A(*Xs))

    for i, X in enumerate(Xs):
      new_Xs = Xs[:i] + (lie_bracket(self.V, X),) + Xs[i + 1:]
      out -= self.A(*new_Xs)

    return out

  def apply_to_point(self, p: Point) -> CovariantTensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    # TODO: IMPLEMENT THIS.  First retrieve set of dense coordinates?
    assert 0, "Not implemented.  Apply this to vector fields first then at a point."

################################################################################################################

def lie_derivative(V: VectorField, x: Union[Map,VectorField,TensorField]) -> Union[Map,VectorField,TensorField]:
  """Compute the Lie derivative L_V(x) where x can be a function,
  vector field, covector field or tensor field

  Args:
    V: Vector field that we're flowing on
    x: The thing to apply the vector field to

  Returns:
    L_V(x)
  """
  if isinstance(x, VectorField):
    return lie_bracket(V, x)

  elif isinstance(x, TensorField) or isinstance(x, CovectorField):
    # Only implementing covariant tensor field lie derivatives
    assert x.type.k == 0
    return LieDerivativeTensorField(V, x)

  elif isinstance(x, Map):
    return V(x)