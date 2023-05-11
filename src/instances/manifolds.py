from functools import partial
from typing import Callable, List, Optional, Generic, Tuple
from src.manifold import Chart
import src.util
import jax.numpy as jnp
from functools import partial
from src.set import *
from src.map import *
from src.vector import *
from src.manifold import *
import src.util as util

__all__ = ["EuclideanManifold",
           "Sphere",
           "RealProjective"]

class EuclideanManifold(Manifold, VectorSpace):
  """R^n

  Attributes:
    dim: Dimensionality
  """
  Element = Coordinate

  def __init__(self, dimension: int, chart: Optional[Callable[[Point,bool],Coordinate]]=None):
    """Create Euclidean space

    Args:
      dimension: Dimension
      chart: Optionally give a prefered choice of coordinates
    """
    self.dimension = dimension
    self.space = EuclideanSpace(dimension=dimension)

    if chart is None:
      self.coordinate_function = lambda x, inverse=False: x
    else:
      self.coordinate_function = chart

    super().__init__(dimension=dimension)
    self.global_chart = Chart(phi=self.coordinate_function, domain=self, image=self.space)

  def get_chart_for_point(self, p: Point) -> Chart:
    """Get a chart for a point

    Args:
      p: Point

    Returns:
      Chart
    """
    return self.global_chart

  def contains(self, p: Point) -> bool:
    """See if p is in this manifold

    Returns:
      True or false
    """
    return p in self.space

  def get_basis(self) -> VectorSpaceBasis:
    """Get a basis of vectors for the vector space

    Returns:
      A list of vector that form a basis for the vector space
    """
    zero = jnp.zeros(self.dimension)
    G = jax.jacobian(self.coordinate_function)(zero)
    J = jnp.linalg.inv(G)
    return [J[:,i] for i in range(self.dimension)]

################################################################################################################

class Sphere(Manifold):
  """An N-sphere

  Attributes:
    dim: Dimensionality of sphere (embedded submanifold of R^(dim+1)).
  """
  Element = Coordinate

  def __init__(self, dim: int):
    """Create a sphere manifold.

    Args:
      dim: Dimension
    """
    self.dim = dim
    super().__init__(dimension=self.dim)

  def get_chart_for_point(self, p: Point) -> Chart:
    """Get a chart to use at point p

    Args:
      The input point

    Returns:
      The chart that contains p in its domain
    """
    north_pole = jnp.zeros(self.dim + 1)
    north_pole = north_pole.at[-1].set(1.0)
    south_pole = -north_pole

    # Check which pole p is closer to
    pole = "south"
    if jnp.linalg.norm(p - north_pole) < jnp.linalg.norm(p - south_pole):
      pole = "north"

    # Only create the charts once
    if hasattr(self, "north_chart"):
      if pole == "north":
        return self.north_chart
      else:
        return self.south_chart

    # Stereographic projection
    def sigma(x, inverse=False):
      # Valid everywhere except north pole
      if inverse == False:
        return x[:-1]/(1 - x[-1])
      else:
        norm_sq = jnp.sum(x**2)
        return jnp.concatenate([2*x, jnp.array([norm_sq - 1.0])])/(norm_sq + 1.0)

    def sigma_tilde(x, inverse=False):
      # Valid everywhere except south pole
      if inverse == False:
        return -sigma(-x)
      else:
        return -sigma(-x, inverse=True)

    class DomainNorth(Sphere):
      def contains(self, p):
        if util.GLOBAL_CHECK == False:
          return True  # Don't bother if we're using diffrax

        if jnp.allclose(jnp.linalg.norm(p), 1.0) == False:
          return False

        # False if p is the north pole
        d_north = jnp.linalg.norm(p - north_pole)
        if d_north < 1e-4:
          return False
        return True

    class DomainSouth(Sphere):
      def contains(self, p):
        if util.GLOBAL_CHECK == False:
          return True  # Don't bother if we're using diffrax

        if jnp.allclose(jnp.linalg.norm(p), 1.0) == False:
          return False

        # False if p is the north pole
        d_south = jnp.linalg.norm(p - south_pole)
        if d_south < 1e-4:
          return False
        return True

    self.north_chart = Chart(phi=sigma, domain=DomainNorth(dim=self.dim), image=EuclideanSpace(dimension=self.dim))
    self.south_chart = Chart(phi=sigma_tilde, domain=DomainSouth(dim=self.dim), image=EuclideanSpace(dimension=self.dim))

    if pole == "north":
      return self.north_chart
    else:
      return self.south_chart

class RealProjective(Manifold):
  """RP^n is the set of 1-d linear subspaces of R^n+1.  If x in R^n+1, then
  [x] is the equivalence class of x, where x ~ y if y = cx for some c in R.

  Attributes:
    dim: Dimensionality
  """
  Element = Coordinate

  def __init__(self, dim: int):
    """Create an RP^n manifold.

    Args:
      dim: Dimension
    """
    self.dim = dim
    super().__init__(dimension=dim)

  def contains(self, p: Point) -> bool:
    """See if p is in this manifold

    Args:
      p: Point

    Returns:
      True or false
    """
    return p.shape == (self.dimension + 1,)

  def get_chart_for_point(self, p: Point) -> Chart:
    """Get a chart to use at point p

    Args:
      The input point

    Returns:
      The chart that contains p in its domain
    """
    # Must exclude origin, but don't worry about this here.
    i = 0

    def phi(x, inverse=False):
      if inverse == False:
        return jnp.concatenate([x[...,:i], x[...,i+1:]], axis=-1)/x[...,i]

      if i == self.dim:
        return jnp.concatenate([x[...,:i], jnp.ones(x.shape[:-1] + (1,))], axis=-1)
      return jnp.concatenate([x[...,:i], 1.0, x[...,i:]], axis=-1)

    U_i = EuclideanSpace(dimension=self.dimension + 1)
    return Chart(phi=phi, domain=U_i, image=EuclideanSpace(dimension=self.dimension))