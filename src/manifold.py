from functools import partial
from typing import Callable, List, Optional, Generic, Tuple
import src.util
import jax.numpy as jnp
from functools import partial
from src.set import *
from src.map import *
import abc
import src.util as util

__all__ = ["Chart",
           "transition_map",
           "Atlas",
           "Manifold",
           "CartesianProductManifold",
           "manifold_cartesian_product"]

class Manifold(Set, abc.ABC):
  """A smooth manifold.  Is described in coordinates using an atlas.

  Attributes:
    atlas: Atlas providing coordinate representation.
  """
  Element = Point

  def __init__(self, dimension: int=None):
    """Create a manifold object.

    Args:
      atlas: Atlas
    """
    super().__init__()
    self.dimension = dimension
    self.atlas = self.get_atlas()

  @abc.abstractmethod
  def get_atlas(self) -> "Atlas":
    """Construct the atlas for a manifold

    Attributes:
      atlas: Atlas providing coordinate representation.
    """
    pass

  def __contains__(self, p: Point) -> bool:
    """Checks to see if p exists in the manifold.  This is the case if
       the point is in the domain of any chart

    Args:
      p: Test point.

    Returns:
      True if p is in the manifold, False otherwise.
    """
    for c in self.atlas.charts:
      if p in c.domain:
        return True
    return False

  @property
  def boundary(self) -> "Manifold":
    """Specify the boundary of the manifold

    Returns:
      The boundary object
    """
    return EmptySet()

  def get_chart_for_point(self, p: Point) -> "Chart":
    """Get a chart to use at point p

    Args:
      The input point

    Returns:
      The chart that contains p in its domain
    """
    return self.atlas.get_chart_for_point(p)

class Chart(Diffeomorphism):
  """A coordinate chart maps between a subset of a manifold and Euclidean space.

  Attributes:
    phi: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  def __init__(self, phi: Callable[[Manifold],Reals], *, domain: Manifold, image: Reals=Reals()):
    """Creates a new function.

    Args:
      f: A function that maps between Euclidean spaces.
      domain: Must be a set of Real numbers.
      image: Must be a set of Real numbers.
    """
    assert isinstance(image, Reals)
    super().__init__(phi, domain=domain, image=image)

  def get_slice_chart(self, mask: Coordinate, consts: Coordinate) -> "Chart":
    """Creates a new Chart. but where some of the coordinates are fixed

    Args:
      mask: A mask so that we know which coordinates to fix
      consts: The values of the coordinates to fix

    Returns:
      A slice chart
    """
    return SliceChart(self, mask, consts)

class SliceChart(Chart):
  """A slice chart is a chart where we have fixed some of the coorindates

  Attributes:
    chart: The original chart.
    mask: A boolean mask telling us which coordinates to fix.
    coords: The coordinates to fix.  Ignore the unmasked coordinates.
  """
  def __init__(self, chart: Chart, mask: jnp.ndarray, coords: Coordinate):
    assert mask.shape == coords.shape
    self.chart = original_chart
    self.mask = mask
    self.coords = coords

    # Construct the domain and codomain
    domain = self.chart.domain
    co_dim = mask.sum()
    image = Reals(dimension=self.chart.image.dimension - co_dim)

    # Create the new chart function
    def phi(p, inverse=False):
      if inverse == False:
        x_coords = self.chart(p)
        return x_coords.at[self.mask]
      else:

        p = jnp.where(self.mask, self.coords, p)
        x = self.chart.inverse(p)
        return x

    super().__init__(phi, domain=domain, image=image)

################################################################################################################

def transition_map(phi: Chart, psi: Chart) -> InvertibleFunction:
  """Create the transition map between the coordinates for each chart.

  Args:
    phi: Chart 1.
    psi: Chart 2.

  Returns:
    psi*phi^{-1}
  """
  return compose(psi, phi.get_inverse())

################################################################################################################

class Atlas(List[Chart]):
  """A collection of charts that describe a manifold

  Attributes:
    charts: A list of charts.
  """
  def __init__(self, charts: List[Chart]):
    """Create an atlas from a bunch of charts

    Args:
      charts: A list of charts
    """
    if isinstance(charts, Chart):
      charts = [charts]

    self.charts = charts

  def get_chart_for_point(self, p: Point) -> Chart:
    """Find the chart associated with a point

    Args:
      p: Point on manifold

    Returns:
      A chart that contains p
    """
    if util.GLOBAL_CHECK:
      for c in self.charts:
        if p in c.domain:
          return c
      assert 0, "p is not on manifold"

    return self.charts[0]

################################################################################################################

class SubImmersion(Manifold):
  """The output of F(M) for a smooth map F and smooth manifold M

  Attributes:
    F: The map
    M: The manifold
  """
  def __init__(self, F: Map, M: Manifold):
    """Creates a new subimmersion

    Args:
      F: The map
      M: The manifold
    """
    assert F.domain == M
    self.F = F
    self.M = M

    charts = []
    for chart_M in self.M.atlas.charts:
      chart = compose(chart_M, self.F.get_inverse())
      charts.append(chart)

    atlas = Atlas(charts)
    super().__init__(atlas, dimension=dimension)

def apply_map_to_manifold(F: InvertibleMap, M: Manifold) -> Manifold:
  """Create the manifold F(M).  F: M -> N

  Args:
    F: A Map object
    M: A Manifold object

  Returns:
    F(M)
  """
  assert hasattr(F, "get_inverse") # F must be invertible
  return SubImmersion(F, M)

################################################################################################################

class CartesianProductManifold(Manifold):
  """Set that is the cartesian product of 2 manifolds.  The chart
  comes from the charts of the sets we're multplying.

  Attributes:
    manifold_a: Manifold a
    manifold_b: Manifold b
  """
  Element = Tuple[Point,Point]

  def __init__(self, a: Manifold, b: Manifold):
    """Create the cartesian product of a and b

    Args:
      a: Set a
      b: Set b
    """
    self.manifold_a = a
    self.manifold_b = b
    dimension = self.manifold_a.dimension + self.manifold_b.dimension
    super().__init__(dimension=dimension)

  def get_atlas(self) -> Atlas:
    """Construct the atlas for the cartesian product.  This involves
    constructing a chart that concatenates the coordinate representation
    of each manifold.

    Attributes:
      atlas: Atlas providing coordinate representation.
    """
    a_dim = self.manifold_a.dimension
    dimension = self.manifold_a.dimension + self.manifold_b.dimension

    # The atlas will have n_charts(a) * n_charts(b) charts
    charts = []
    for chart_a in self.manifold_a.atlas.charts:
      for chart_b in self.manifold_b.atlas.charts:

        def phi(x, inverse=False):
          if inverse == False:
            p, v = x
            p_coords, v_coords = chart_a(p), chart_b(v)
            return jnp.concatenate([p_coords, v_coords], axis=-1)
          else:
            p_coords, v_coords = x[:a_dim], x[a_dim:]
            p, v = chart_a.inverse(p_coords), chart_b.inverse(v_coords)
            return (p, v)

        domain = set_cartesian_product(chart_a.domain, chart_b.domain)
        new_chart = Chart(phi=phi, domain=domain, image=Reals(dimension=dimension))
        charts.append(new_chart)

    return Atlas(charts)

def manifold_cartesian_product(a: Manifold, b: Manifold) -> CartesianProductManifold:
  """The cartesian product of two manifolds

  Args:
    a: Manifold A
    b: Manifold B

  Returns:
    Cartesian product AxB
  """
  return CartesianProductManifold(a, b)

################################################################################################################
