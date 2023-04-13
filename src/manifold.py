from functools import partial
from typing import Callable, List, Optional, Generic, Tuple
import src.util
import jax.numpy as jnp
from functools import partial
from src.set import *
from src.map import *
import abc
import src.util as util
import itertools

__all__ = ["Chart",
           "SliceChart",
           "Atlas",
           "Manifold",
           "SubImmersion",
           "apply_map_to_manifold",
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
      dimension: Dimension of the manifold
    """
    super().__init__()
    self.dimension = dimension
    self.atlas = self.get_atlas()

  def __str__(self) -> str:
    """The string representation of this manifold

    Returns:
      A string
    """
    return f"{type(self)}(dimension={self.dimension})"

  @abc.abstractmethod
  def get_atlas(self) -> "Atlas":
    """Construct the atlas for a manifold

    Args:
      atlas: Atlas providing coordinate representation.
    """
    pass

  def contains(self, p: Point) -> bool:
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
  def __init__(self, phi: Callable[[Manifold],Reals], *, domain: Manifold, image: Reals):
    """Creates a new function.

    Args:
      f: A function that maps between Euclidean spaces.
      domain: Must be a set of Real numbers.
      image: Must be a set of Real numbers.
    """
    assert isinstance(image, Reals)
    super().__init__(phi, domain=domain, image=image)

  def get_slice_chart(self, mask: Coordinate) -> "Chart":
    """Creates a new slice chart.

    Args:
      mask: A mask so that we know which coordinates to fix

    Returns:
      A slice chart
    """
    return SliceChart(self, mask)

class SliceChart(Chart):
  """A slice chart is a chart where we have fixed some of the coorindates

  Attributes:
    chart: The original chart.
    mask: A boolean mask telling us which coordinates to fix.
  """
  def __init__(self, chart: Chart, mask: jnp.ndarray):
    self.chart = chart
    self.mask = mask
    self.coords = None

    # Construct the domain and codomain
    domain = self.chart.domain
    co_dim = mask.sum()
    from src.instances.manifolds import EuclideanManifold
    image = EuclideanManifold(dimension=self.chart.image.dimension - co_dim)

    # Create the new chart function
    def phi(p, inverse=False):
      if inverse == False:
        # TODO: MAKE A SAFE VERSION OF THIS
        self.coords = self.chart(p)
        return self.coords[self.mask]
      else:
        p = jnp.where(self.mask, self.coords, p)
        x = self.chart.inverse(p)
        return x

    super().__init__(phi, domain=domain, image=image)

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

  def __str__(self):
    return str(self.chart)

  def __repr__(self):
    return str(self.charts)

  def get_chart_for_point(self, p: Point) -> Chart:
    """Find the chart associated with a point

    Args:
      p: Point on manifold

    Returns:
      A chart that contains p
    """
    for c in self.charts:
      if p in c.domain:
        return c
    assert 0, "p is not on manifold"

################################################################################################################

class SubImmersion(Manifold):
  """The output of F(M) for a smooth map F and smooth manifold M

  Attributes:
    F: The map
    M: The manifold
  """
  def __init__(self, F: InvertibleMap, M: Manifold):
    """Creates a new subimmersion

    Args:
      _F: _F should be a regular python function with an inverse
      M: The manifold
    """
    assert F.domain == M
    self.F = F
    self.M = M
    super().__init__(dimension=self.M.dimension)

    self.F = InvertibleMap(self.F, domain=self.M, image=self)

  def get_atlas(self):
    """Return the atlas

    Returns:
      Atlas object
    """
    charts = []
    for chart_M in self.M.atlas.charts:
      chart = compose(chart_M, self.F.get_inverse())
      charts.append(chart)

    return Atlas(charts)

def apply_map_to_manifold(F: InvertibleMap, M: Manifold) -> Manifold:
  """Create the manifold F(M).  F: M -> N

  Args:
    F: An invertible map
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
    Ms: The manifolds that we're multiplying
  """
  Element = List[Point]

  def __init__(self, *Ms: List[Manifold]):
    """Create the cartesian product of manifolds

    Args:
      Ms: A list of manifolds
    """
    self.Ms = Ms
    self.dimensions = [M.dimension for M in self.Ms]
    super().__init__(dimension=sum(self.dimensions))

  def get_atlas(self) -> Atlas:
    """Construct the atlas for the cartesian product.  This involves
    constructing a chart that concatenates the coordinate representation
    of each manifold.

    Attributes:
      atlas: Atlas providing coordinate representation.
    """
    new_charts = []
    for charts in itertools.product(*[M.atlas.charts for M in self.Ms]):

      def phi(x, inverse=False):
        if inverse == False:
          assert len(x) == len(charts)
          coords = [chart(_x) for _x, chart in zip(x, charts)]
          return jnp.concatenate(coords, axis=-1)
        else:
          # Split the coordinates
          cum_dims = list(itertools.accumulate(self.dimensions))
          split_indices = list(zip([0] + cum_dims[:-1], cum_dims))
          coords = [x[start:end] for start, end in split_indices]
          assert len(coords) == len(charts)
          return [chart.inverse(coord) for coord, chart in zip(coords, charts)]

      domain = set_cartesian_product(*self.Ms)
      new_chart = Chart(phi=phi, domain=domain, image=Reals(dimension=self.dimension))
      new_charts.append(new_chart)

    return Atlas(new_charts)

def manifold_cartesian_product(*Ms: List[Manifold]) -> CartesianProductManifold:
  """The cartesian product of manifolds

  Args:
    Ms: A list of manifolds

  Returns:
    Cartesian product of Ms
  """
  return CartesianProductManifold(*Ms)

################################################################################################################
