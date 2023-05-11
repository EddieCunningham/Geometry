from typing import List, Optional, Generic, Tuple, TypeVar
import jax.numpy as jnp
from src.set import *
from src.map import *
import src.util as util
import abc
from simple_pytree import Pytree

__all__ = ["Coordinate",
           "Matrix",
           "InvertibleMatrix",
           "EuclideanSpace",
           "Vector",
           "VectorSpace",
           "VectorSpaceBasis"]

"""Coordinate.  Will represent a point on Euclidean space and can be
  implemented using numpy arrays.  Important to make this distinction
  because we will always want to do computations in Euclidean space.
"""
Coordinate = TypeVar("Coordinate", bound=jnp.array)

class Matrix(Generic[Coordinate]):
  pass

class InvertibleMatrix(Generic[Coordinate]):
  pass

################################################################################################################

class Vector(Generic[Point], Pytree):
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
    assert a in EuclideanSpace(dimension=1)
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

class EuclideanSpace(Set):
  """Set of real numbers.  We can do computations here.

  Attributes:
    dimension: dimension of this space.
    membership_function: Function that determines membership.
  """
  Element = Vector

  def __init__(self, dimension: int):
    """Creates a new Euclidean space.  Can optionally set the dimension as well.

    Args:
      dimension: An integer representing the dimensionality of the space.
      membership_function: Function that determines membership.
    """
    assert dimension is not None
    self.dimension = dimension
    super().__init__()

  def __str__(self) -> str:
    """The string representation of this manifold

    Returns:
      A string
    """
    return f"{type(self)}(dimension={self.dimension})"

  def contains(self, p: Coordinate) -> bool:
    """Checks to see if p exists in this set and is real.

    Args:
      p: Test point.

    Returns:
      True if p is in the set and real, False otherwise.
    """
    if isinstance(p, float) and ((self.dimension is None) or (self.dimension <= 1)):
      return True

    # Must be a coordinate type
    if isinstance(p, jnp.ndarray) == False:
      return False

    # Must be a vector or a scalar
    if p.ndim > 1:
      return False

    # Optionally, must have a fixed dimensionality
    if self.dimension is not None:
      if self.dimension > 1:
        if p.shape[-1] != self.dimension:
          return False
      else:
        if len(p.shape) == 0:
          pass # This is ok
        elif p.shape[-1] != self.dimension:
          return False

    return super().contains(p)

  def __eq__(self, other: Set) -> bool:
    """Checks to see if another set is the set of reals.
    Because we construct these often, it doesn't make
    sense to compare the instances to each other.

    Args:
      other: A set

    Returns:
      boolean
    """
    from src.instances.manifolds import EuclideanManifold

    if isinstance(other, EuclideanManifold):
      if self.dimension is None:
        return other.dimension <= 1
      return self.dimension == other.dimension

    type_check = isinstance(other, EuclideanSpace)

    if other.dimension is None:
      if self.dimension:
        dim_check = self.dimension <= 1
      else:
        dim_check = True # Both dimensions are None
    else:
      dim_check = self.dimension == other.dimension
    return dim_check and type_check

  def get_basis(self) -> "VectorSpaceBasis":
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

class VectorSpace(EuclideanSpace, abc.ABC):
  """An vector space will need a choice of coordinate function.

  Attributes:
    dim: Dimensionality
  """
  Element = Vector

  def __init__(self, dimension: int):
    """Create Euclidean space

    Args:
      dimension: Dimension
      chart: A chart for the vector space (choose a basis)
    """
    super().__init__(dimension=dimension)

  def contains(self, v: Vector) -> bool:
    """Checks to see if v is a part of this vector space

    Args:
      v: An input vector

    Returns:
      True or false
    """
    type_check = isinstance(v, self.Element)
    space_check = v.V == self
    coordinate_check = super().contains(v.x)
    return type_check and space_check and coordinate_check

  def get_basis(self) -> "VectorSpaceBasis":
    """Get a basis of vectors for the vector space

    Returns:
      A list of vector that form a basis for the vector space
    """
    return super().get_basis()

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

  def __len__(self) -> int:
    """Get the number of elements in this basis

    Returns:
      len(self)
    """
    return len(self.basis)

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
    assert a in EuclideanSpace(dimension=1)

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
