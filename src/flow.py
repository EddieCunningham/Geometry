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
from src.lie_group import *
from src.vector_field import *
from src.bundle import *
import src.util as util
import diffrax

__all__ = ["IntegralCurve",
           "Flow",
           "FlowInducedByGroupAction",
           "get_infinitesmal_generator_map",
           "TimeDependentFlow"]

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
    super().__init__(self.__call__, domain=Reals(), image=self.manifold)

  def __call__(self, t: Coordinate) -> Point:
    """Evaluate the integral curve at t

    Args:
      t: Time

    Returns:
      Value at t
    """
    sign = jnp.where(t < 0.0, -1.0, 1.0)

    # Will be using the same chart for all points on the trajectory!
    # TODO: Integrate using dynamic charts
    chart = self.manifold.get_chart_for_point(self.p0)
    p0_hat = chart(self.p0)

    # Integrate on the manifold in coordinate space
    def f(t, p_hat, args):
      # Get the point corresponding to the current coordintes
      p = chart.inverse(p_hat)

      # Evaluate the vector field and get coordinates of the manifold
      Vp = self.V(p)

      # Swap the sign if needed
      Vp = sign*Vp

      # Get the coordinate representation of this vector using
      # the chart coordinates
      return Vp.get_coordinates(chart)

    term = diffrax.ODETerm(f)
    solver = diffrax.Dopri5()
    solution = diffrax.diffeqsolve(term, solver, t0=0.0, t1=sign*t, dt0=0.001, y0=p0_hat)
    z = solution.ys[0]
    return chart.inverse(z)

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

  def infinitesmal_generator(self, p: Point) -> TangentVector:
    """Get the tangent vector for p

    Args:
      p: An input point.

    Returns:
      V(p)
    """
    return self.V(p)

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

class FlowInducedByGroupAction(Flow):
  """Given a manifold M and a group G that has a right group action on M, this flow
  is defined by (t,p) |-> p*exp(tX) where X is an element of the Lie algebra of G.

  Attributes:
    manifold: The manifold that the flow is defined on
    G: The group with a right group action over M
    X: An element of the Lie algebra of G that determines one parameter subgroup
    right_action: Whether or not to use the right group action on the manifold
  """
  def __init__(self, M: Manifold, G: LieGroup, X: TangentVector, right_action: Optional[bool]=True):
    self.manifold = M
    self.G = G
    self.lieG = self.G.get_lie_algebra()
    self.X = X
    self.ops = self.lieG.get_one_parameter_subgroup(self.X)

    self.right_action = right_action

  def __call__(self, t: Coordinate, p: Point) -> Point:
    """Applies the function on p.

    Args:
      t: Input time
      p: An input point.

    Returns:
      f(p)
    """
    g = self.ops(t)
    if self.right_action:
      return self.G.right_action_map(g, self.manifold)(p)
    else:
      return self.G.left_action_map(g, self.manifold)(p)

  def infinitesmal_generator(self, p: Point) -> TangentVector:
    """Get the tangent vector for p.  This is d(theta^(p))_e(X_e)
    Basially pushforward of X_e through the map theta^(p): h |--> p*h

    Args:
      p: An input point.

    Returns:
      Xhat_p
    """
    if self.right_action:
      dtheta_p_e = self.G.right_orbit_map(p, self.manifold).get_differential(self.G.e)
    else:
      dtheta_p_e = self.G.left_orbit_map(p, self.manifold).get_differential(self.G.e)
    Xe = self.X(self.G.e)
    return dtheta_p_e(Xe)

def get_infinitesmal_generator_map(G: LieGroup, M: Manifold, right_action: Optional[bool]=True) -> Map["LeftInvariantVectorField",VectorField]:
  """Get the tangent vector for p.  This is d(theta^(p))_e(X_e)

  Args:
    p: An input point.

  Returns:
    V(p)
  """
  assert isinstance(G, LieGroup)
  def theta_hat(X):
    flow = FlowInducedByGroupAction(M, G, X, right_action=right_action)
    class InfinitesmalGeneratorVectorField(VectorField):
      def apply_to_point(self, p: Point) -> TangentVector:
        return flow.infinitesmal_generator(p)

    return InfinitesmalGeneratorVectorField(M)
  return Map(theta_hat, domain=G.get_lie_algebra(), image=TangentBundle(M))

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
      return Vp.get_coordinates(IdentityMap(manifold=self.manifold))

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
