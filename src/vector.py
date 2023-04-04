from functools import partial
from typing import Callable, List, Optional, Generic, Tuple
import src.util
import jax.numpy as jnp
from functools import partial
from src.set import *
from src.map import *
from src.manifold import *
from src.instances.manifolds import EuclideanManifold
import src.util as util

__all__ = ["Vector",
           "VectorSpace",
           "VectorSpaceBasis",
           "VectorBundle"]

class Vector(Generic[Point]):
  """A vector in some vector space

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    V: The vector space
  """
  x: Coordinate
  V: "VectorSpace"

  def __init__(self, x: Coordinate, V: "VectorSpace"):
    """Creates a new vector.

    Args:
      x: The coordinates of the vector in the basis induced by a chart.
      V: The vector space that the vector lives on
    """
    assert x.ndim == 1
    self.x = x
    self.V = V
    assert x.shape[0] == self.V.dimension

  def __add__(self, Y: "Vector") -> "Vector":
    """Add two vectors together

    Args:
      Y: Another vector

    Returns:
      Xp + Yp
    """
    # Must be the same type
    assert isinstance(Y, type(self))

    # Can only add if they are a part of the same vector space
    assert self.V == Y.V

    # Add the coordinates together
    return type(self)(self.x + Y.x, self.V)

  def __radd__(self, Y: "Vector") -> "Vector":
    """Add Y from the right

    Args:
      Y: Another vector

    Returns:
      X + Y
    """
    # Must be the same type
    assert isinstance(Y, type(self))

    return self + Y

  def __rmul__(self, a: float) -> "Vector":
    """Vectors support scalar multiplication

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    assert a in Reals(dimension=1)
    return type(self)(self.x*a, self.V)

  def __sub__(self, Y: "Vector") -> "Vector":
    """Subtract Y from this vector

    Args:
      Y: Another vector

    Returns:
      X - Y
    """
    # Must be the same type
    assert isinstance(Y, type(self))

    return self + -1.0*Y

  def __neg__(self):
    """Negate this vector

    Returns:
      Negative of this vector
    """
    return -1.0*self

################################################################################################################

class VectorSpace(EuclideanManifold):
  """An vector space will need a choice of coordinate function.

  Attributes:
    dim: Dimensionality
  """
  Element = Vector
  chart: Chart

  def __init__(self, dimension: int):
    """Create Euclidean space

    Args:
      dimension: Dimension
      chart: A chart for the vector space (choose a basis)
    """
    super().__init__(dimension=dimension)

  def __contains__(self, v: Vector) -> bool:
    """Checks to see if v is a part of this vector space

    Args:
      v: An input vector

    Returns:
      True or false
    """
    type_check = isinstance(v, self.Element)
    space_check = v.V == self
    return type_check and space_check

  def get_atlas(self):
    """Return the atlas

    Returns:
      Atlas object
    """

    # We're already using coordinates to represent vectors,
    # we don't need to do anything special.
    def chart_fun(v, inverse=False):
      if inverse == False:
        return v.x
      else:
        # Create an object of the elements that this vector
        # space contains
        return self.Element(v, self)

    self.chart = Chart(chart_fun, domain=self, image=Reals(dimension=self.dimension))
    return Atlas([self.chart])

  def get_basis(self) -> List[Vector]:
    """Get a basis of vectors for the vector space

    Returns:
      A list of vector that form a basis for the vector space
    """
    eye = jnp.eye(self.dimension)
    basis = []
    for i in range(self.dimension):
      v = self.Element(eye[i], self)
      basis.append(v)
    return VectorSpaceBasis(basis)

################################################################################################################

class VectorSpaceBasis():
  """A list of vectors that forms a basis for the vector space

  Attributes:
    basis: A list of tangent vectors
    V: The vector space
  """
  def __init__(self, basis_vectors: List[Vector], V: VectorSpace):
    """Create a new vector space basis object

    Args:
      basis_vectors: A list of basis vectors
      V: The vector space that this is a basis of
    """
    # Ensure that each vector is a part of the same tangent space
    assert sum([X.V != V for X in basis_vectors]) == 0
    self.basis = basis_vectors
    self.V = V
    assert len(self.basis) == self.V.dimension

  def __getitem__(self, index: int) -> Vector:
    """Get an item of the basis

    Args:
      index: Which basis vector to get

    Returns:
      basis[index]
    """
    return self.basis[index]

  def __setitem__(self, index: int, v: Vector):
    """Change an element of the basis

    Args:
      index: Which basis vector to set
      v: A vector
    """
    assert isinstance(v, self.V.Element)
    self.basis[index] = v

  def __add__(self, Y: "VectorSpaceBasis") -> "VectorSpaceBasis":
    """Add two tangent space bases.

    Args:
      Y: Another tangent basis

    Returns:
      Xp + Yp
    """
    # Must be the same type
    assert isinstance(Y, type(self))

    # Can only add if they are a part of the same tangent space
    assert self.V == Y.V

    new_basis = [u + v for u, v in zip(self.basis, Y.basis)]
    return type(self)(new_basis, self.V)

  def __radd__(self, Y: "VectorSpaceBasis") -> "VectorSpaceBasis":
    """Add Y from the right

    Args:
      Y: Another tangent basis

    Returns:
      X + Y
    """
    # Must be the same type
    assert isinstance(Y, type(self))

    return self + Y

  def __rmul__(self, a: float) -> "VectorSpaceBasis":
    """Tangent basis support scalar multiplication

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    assert a in Reals(dimension=1)

    new_basis = [a*v for v in self.basis]
    return type(self)(new_basis, self.V)

  def __sub__(self, Y: "VectorSpaceBasis") -> "VectorSpaceBasis":
    """Subtract Y from this basis

    Args:
      Y: Another tangent basis

    Returns:
      X - Y
    """
    # Must be the same type
    assert isinstance(Y, type(self))
    return self + -1.0*Y

  def __neg__(self):
    """Negate this basis

    Returns:
      Negative of this basis
    """
    return -1.0*self

################################################################################################################

class VectorBundle():
# class VectorBundle(FiberBundle, abc.ABC):
  """A fiber bundle where the fiber is R^k.  This is still an abstract
  class because we need a projection map and local trivialization.

  Attributes:
    F: The fiber F.
    M: The base space.
    E: The total space which looks looks like MxF around points.
    pi: Projection map from E to M.
  """
  Element = Tuple[Point,Coordinate]

  def __init__(self, M: Manifold, k: int):
    """Creates a new vector bundle object.

    Args:
      M: The manifold.
    """
    super().__init__(M, EuclideanManifold(dimension=k))
