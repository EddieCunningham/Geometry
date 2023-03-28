from functools import partial
from typing import Callable, List, Optional, Tuple
import src.util
from functools import partial
from copy import deepcopy
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
from src.tangent import *
from src.section import *
from src.vector_field import *
import src.util as util
import diffrax

__all__ = ["IntegralCurve",
           "Flow"]

class IntegralCurve(Map[Coordinate,Point]):
  """An integral curve is a 1d curve whose tangent space
  is given by a vector field.

  Attributes:
    p0: Point at time t=0
    V: The vector field that generates the curve
    manifold: The manifold that the curve lives on
    domain: Reals
    image: Manifold
  """
  def __init__(self, p0: Point, V: VectorField):
    """Creates a new integral curve

    Args:
      p0: Point at time t=0
      V: The vector field that generates the curve
    """
    self.p0 = p0
    self.V = V
    self.manifold = self.V.manifold
    self.domain = Reals()
    self.image = self.manifold

  def __call__(self, t: Coordinate) -> Point:
    """Evaluate the integral curve at t

    Args:
      t: Time

    Returns:
      Value at t
    """
    # Integrate on the manifold starting at t=0
    def f(t, p, args):
      # Evaluate the vector field and get Euclidean coordinates
      Vp = self.V(p)
      return Vp(IdentityMap(manifold=self.manifold))

    term = diffrax.ODETerm(f)
    solver = diffrax.Dopri5()
    solution = diffrax.diffeqsolve(term, solver, t0=0.0, t1=t, dt0=0.001, y0=self.p0)
    z = solution.ys[0]
    return z

  def as_map(self) -> Map:
    """Return this function as a map.  Makes composition easier because
    __init__ function

    Returns:
      Map object wrapper
    """
    return Map(self, domain=self.domain, image=self.image)

################################################################################################################

class Flow(Map[Tuple[Coordinate,Point],Point]):
  """A flow is a continuous left R action on M.
  theta: R x M -> M s.t. theta(t, theta(s, p)) = theta(s + t, p) and theta(0, p) = 0

  Attributes:
    manifold: The manifold that the flow is defined on
    V: The generating vector field
  """
  def __init__(self, M: Manifold, V: VectorField):
    """Creates a new flow

    Args:
      M: The manifold that the flow is defined of
      V: The vector field that is the infinitesmal generator of theta
    """
    self.manifold = M
    self.V = V

  def __call__(self, t: Coordinate, p: Point) -> Point:
    """Applies the function on p.

    Args:
      t: Input time
      p: An input point.

    Returns:
      f(p)
    """
    return IntegralCurve(p, self.V)(t)

  def get_theta_p(self, p: Point) -> Callable[[Coordinate],Point]:
    """Return the map theta^{(p)}(t) = theta(t, p)

    Args:
      p: The point to fix

    Returns:
      A map from R to the orbit of p under the left R-action of M
    """
    def theta_p(t: Coordinate) -> Point:
      return self(t, p)
    return Map(theta_p, domain=Reals(), image=self.manifold)

  def get_theta_t(self, t: Coordinate) -> Callable[[Point],Point]:
    """Return the map theta_t(p) = theta(t, p)

    Args:
      t: Time to fix

    Returns:
      Map with a fixed time
    """
    def theta_t(p: Point) -> Point:
      return self(t, p)
    return Map(theta_t, domain=self.manifold, image=self.manifold)

################################################################################################################

class TimeDependentFlow(Map[Tuple[Coordinate,Coordinate,Point],Point]):
  """A flow that comes from a time dependent vector field is psi(t, t0, p)

  Attributes:
    manifold: The manifold that the flow is defined on
    V: The generating vector field
  """
  def __init__(self, M: Manifold, V: TimeDependentVectorField):
    """Creates a new time dependent flow

    Args:
      M: The manifold that the flow is defined of
      V: The vector field that is the infinitesmal generator of phi
    """
    self.manifold = M
    self.V = V

  def __call__(self, t0: Coordinate, t: Coordinate, p: Point) -> Point:
    """Applies the function on p.

    Args:
      t: Input time
      p: An input point.

    Returns:
      f(p)
    """
    # Integrate on the manifold starting at t=0
    def f(t, p, args):
      # Evaluate the vector field and get Euclidean coordinates
      Vp = self.V(t, p)
      return Vp(IdentityMap(manifold=self.manifold))

    term = diffrax.ODETerm(f)
    solver = diffrax.Dopri5()
    solution = diffrax.diffeqsolve(term, solver, t0=t0, t1=t, dt0=0.001, y0=p)
    z = solution.ys[0]
    return z

  def get_psi_p(self, p: Point) -> Callable[[Coordinate],Point]:
    """Return the map phi^{(p)}(t) = phi(t0, t, p)

    Args:
      p: The point to fix

    Returns:
      A map from R to the orbit of p under the left R-action of M
    """
    def psi_p(t0: Coordinate, t: Coordinate) -> Point:
      return self(t0, t, p)
    return Map(psi_p, domain=Reals(), image=self.manifold)

  def get_psi_t(self, t0: Coordinate, t: Coordinate) -> Callable[[Point],Point]:
    """Return the map psi_t(p) = phi(t, p)

    Args:
      t: Time to fix

    Returns:
      Map with a fixed time
    """
    def psi_t(p: Point) -> Point:
      return self(t0, t, p)
    return Map(psi_t, domain=self.manifold, image=self.manifold)

################################################################################################################

def lie_derivative(V: VectorField, W: VectorField):
  """Compute the Lie derivative L_V(W) = [V, W]

  Args:
    V: Vector field 1
    W: Vector field 2

  Returns:
    L_V(W) = [V, W]
  """
  return lie_bracket(V, W)