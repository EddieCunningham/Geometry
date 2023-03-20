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
from src.vector_field import *
from src.section import *
import src.util as util

__all__ = ["AutonomousVectorField",
           "AutonomousFrame",
           "SimpleTimeDependentVectorField"]

class StandardBasis(VectorField):
  pass

class AutonomousVectorField(VectorField):
  """A vector field where we have a function to give us
  the coordinates of tangent vectors using whatever basis
  we get from charts of the manifold.  This is autonomous
  so doesn't depend on t.

  Attribues:
    vf: The function that gives us points
    M: Manifold
  """
  def __init__(self, vf: Callable[[Coordinate], Coordinate], M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      vf: Vector field coordinate function.
      M: The base manifold.
    """
    self.vf = vf
    self.manifold = M

    super().__init__(self.manifold)

    # Test that the vf is compatible with the manifold
    x = jnp.zeros(self.manifold.dimension)
    out = vf(x)
    assert out.shape == x.shape

  def __call__(self, p: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the vector field coordinates at p
    v_coords = self.vf(p_coords)

    # Construct the tangent vector
    TpM = TangentSpace(p, self.manifold)
    return TangentVector(v_coords, TpM)

################################################################################################################

class AutonomousFrame(Frame):
  """A frame implemented as the Jacobian of a function

  Attribues:
    vf: The function that gives us points
    M: Manifold
  """
  def __init__(self, vf: Callable[[Coordinate], Coordinate], M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      vf: Vector field coordinate function.
      M: The base manifold.
    """
    self.vf = jax.jacobian(vf)
    self.manifold = M

    super().__init__(self.manifold)

    # Test that the vf is compatible with the manifold
    x = jnp.zeros((self.manifold.dimension, self.manifold.dimension))
    out = vf(x)
    assert out.shape == x.shape

  def __call__(self, p: Point) -> TangentBasis:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the vector field coordinates at p
    v_coords = self.vf(p_coords)

    TpM = TangentSpace(p, self.manifold)
    return MatrixColsToList(TpM)(v_coords)

################################################################################################################

class SimpleTimeDependentVectorField(TimeDependentVectorField):
  # Construct a vector field using the charts for a manifold
  def __init__(self, vf, M: Manifold):
    self.vf = vf
    self.manifold = M

    # Test that the vf is compatible with the manifold
    x = jnp.zeros(self.manifold.dimension)
    out = vf(0.0, x)
    assert out.shape == x.shape

  def __call__(self, t: Coordinate, p: Point) -> TangentVector:
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the vector field coordinates at p
    v_coords = self.vf(t, p_coords)

    # Construct the tangent vector
    TpM = TangentSpace(p, self.manifold)
    return TangentVector(v_coords, TpM)