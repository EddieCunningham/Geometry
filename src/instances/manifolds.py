from functools import partial
from typing import Callable, List, Optional, Generic, Tuple
import src.util
import jax.numpy as jnp
from functools import partial
from src.set import *
from src.map import *
from src.manifold import *
import src.util as util

__all__ = ["EuclideanManifold",
           "Vector",
           "VectorSpace",
           "Sphere",
           "RealProjective"]

class EuclideanManifold(Manifold, Reals):
  """R^n

  Attributes:
    dim: Dimensionality
  """
  Element = Coordinate

  def __init__(self, dimension: int):
    """Create Euclidean space

    Args:
      dimension: Dimension
    """
    self.dimension = dimension
    self.space = Reals(dimension=dimension)
    super().__init__(dimension=dimension)

  def get_atlas(self):
    """Return the atlas

    Returns:
      Atlas object
    """
    return Atlas([Chart(phi=lambda x, inverse=False: x, domain=self.space, image=self.space)])

  def __contains__(self, p: Point) -> bool:
    """See if p is in this manifold

    Returns:
      True or false
    """
    return p in self.space

  def add(self, u: Point, v: Point) -> Point:
    """We can add 2 vectors together

    Args:
      u: Vector 1
      v: Vector 2

    Returns:
      u + v
    """
    return u + v

  def mult(self, a: float, v: Point) -> Point:
    """We can multiply a scalar with a vector

    Args:
      a: Scalar
      v: Vector 2

    Returns:
      a*v
    """
    return a*v

  def to_vector_space(self, basis: Chart) -> "VectorSpace":
    """Return a vector space.  Vector spaces are isomorphic to R^n
    because they just have a different basis

    Args:
      basis: A chart function that we can use to get basis vectors

    Returns:
      V
    """
    return VectorSpace(dimension=self.dimension, basis=basis)

################################################################################################################

class Vector(Generic[Point]):
  pass

class VectorSpace(EuclideanManifold):
  """An vector space will need a choice of coordinate function.

  Attributes:
    dim: Dimensionality
  """
  Element = Vector

  def __init__(self, dimension: int, chart: Chart):
    """Create Euclidean space

    Args:
      dimension: Dimension
      chart: A chart for the vector space (choose a basis)
    """
    self.dimension = dimension
    self.space = Reals(dimension=dimension)
    self.chart = chart

    # Skip the Euclidean manifold init
    super().__init__(dimension=dimension)

  def get_atlas(self):
    """Return the atlas

    Returns:
      Atlas object
    """
    return Atlas([self.chart])

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

  def get_atlas(self):
    charts = []

    north_pole = jnp.zeros(self.dim + 1)
    north_pole = north_pole.at[-1].set(1.0)
    south_pole = -north_pole

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

    class DomainNorth(Reals):
      def __contains__(self, p):
        if jnp.allclose(jnp.linalg.norm(p), 1.0) == False:
          return False

        # False if p is the north pole
        d_north = jnp.linalg.norm(p - north_pole)
        if d_north < 1e-4:
          return False
        return True

    class DomainSouth(Reals):
      def __contains__(self, p):
        if jnp.allclose(jnp.linalg.norm(p), 1.0) == False:
          return False

        # False if p is the north pole
        d_south = jnp.linalg.norm(p - south_pole)
        if d_south < 1e-4:
          return False
        return True

    north_chart = Chart(phi=sigma, domain=DomainNorth(), image=Reals(dimension=self.dim))
    south_chart = Chart(phi=sigma_tilde, domain=DomainSouth(), image=Reals(dimension=self.dim))

    return Atlas([north_chart, south_chart])

class RealProjective(Manifold):
  """RP^n is the set of 1-d linear subspaces of R^n+1.

  Attributes:
    dim: Dimensionality
  """
  def __init__(self, dim: int):
    """Create an RP^n manifold.

    Args:
      dim: Dimension
    """
    self.dim = dim
    super().__init__(dimension=dim)

  def get_atlas(self):

    charts = []

    for i in range(self.dim + 1):
      # Must exclude origin, but don't worry about this here.
      U_i = Reals(dimension=self.dim + 1)

      def phi(x, inverse=False):
        if inverse == False:
          return jnp.concatenate([x[...,:i], x[...,i+1:]], axis=-1)/x[...,i]

        if i == self.dim:
          return jnp.concatenate([x[...,:i], jnp.ones(x.shape[:-1] + (1,))], axis=-1)
        return jnp.concatenate([x[...,:i], 1.0, x[...,i:]], axis=-1)

      charts.append(Chart(phi=phi, domain=U_i, image=Reals()))

    return Atlas(charts)