from functools import partial
from typing import Callable, List, TypeVar, Generic, Tuple, Union, Iterable
import src.util
from functools import partial, reduce
import jax.numpy as jnp
import jax
from src.set import *
from src.vector import *
import abc
import src.util as util

__all__ = ["Input",
           "Output",
           "Map",
           "IdentityMap",
           "LinearMap",
           "InvertibleLinearMap",
           "MultlinearMap",
           "ProjectionMap",
           "Function",
           "InvertibleFunction",
           "InvertibleMap",
           "Diffeomorphism",
           "compose",
           "curry"]

Input = TypeVar("Input", bound=Point)
Output = TypeVar("Output", bound=Point)

class Map(Generic[Input,Output]):
  """A map takes a point on some domain and maps it to a point on its image.

  Attributes:
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  def __init__(self, f: Callable[[Set],Set], *, domain: Set, image: Set):
    """Creates a new Map.

    Args:
      f: A function that maps between Euclidean spaces.
      domain: Must be a set of Real numbers.
      image: Must be a set of Real numbers.
    """
    self.f = f
    self.domain = domain
    self.image = image

  def __str__(self) -> str:
    """The string representation of this map

    Returns:
      A string
    """
    return f"f: {self.f}, domain: {str(self.domain)}, image: {str(self.image)}"

  def __call__(self, p: Point) -> Point:
    """Applies the function on p.

    Args:
      p: An input point.

    Returns:
      f(p)
    """
    assert p in self.domain
    q = self.f(p)
    assert q in self.image
    return q

  def get_coordinate_map(self, p: Input) -> "Map":
    """Return the coordinate map corresponding to F.
       Let F: M-> N and let phi and psi be charts for M and N.
       Then the coordinate map is psi*F*phi^{-1}

    Args:
      None

    Returns:
      The coordinate map
    """
    M, N = self.domain, self.image
    q = self(p)
    phi = M.get_chart_for_point(p)
    psi = N.get_chart_for_point(q)
    return compose(psi, self, phi.get_inverse())

  def get_differential(self, p: Point) -> "Differential":
    """Get the differential of this map at p

    Args:
      p: Point to evaluate rank at

    Return:
      dFp
    """
    from src.tangent import Differential
    return Differential(self, p)

  def get_pullback(self, p: Point) -> "Pullback":
    """Get the pullback of this map at p

    Args:
      p: Point to evaluate rank at

    Return:
      dFp^*
    """
    from src.cotangent import Pullback
    return Pullback(self, p)

  def get_tensor_pullback(self, p: Point, tensor_type: "TensorType") -> "PullbackOfTensor":
    """Get the pullback of this map at p.
    # TODO: MERGE THIS WITH THE REGULAR PULLBACK

    Args:
      p: Point to evaluate rank at

    Return:
      dFp^*
    """
    from src.tensor import PullbackOfTensor
    return PullbackOfTensor(self, p, tensor_type)

  def rank(self, p: Point) -> int:
    """Returns the rank of the map at p

    Args:
      p: Point to evaluate rank at

    Return:
      Rank of map
    """
    dFp = self.get_differential(p)
    return jnp.linalg.matrix_rank(dFp.get_coordinates())

  def is_submersion(self, p: Point) -> bool:
    """Check to see if the map is a submersion at p

    Args:
      p: Point to evaluate rank at

    Return:
      Whether or not is submersion
    """
    return self.rank(p) == self.image.dimension

  def is_immersion(self, p: Point) -> bool:
    """Check to see if the map is a immersion at p

    Args:
      p: Point to evaluate rank at

    Return:
      Whether or not is immersion
    """
    return self.rank(p) == self.domain.dimension

  def is_diffeomorphism(self, p: Point) -> bool:
    """Check to see if the map is a diffeomorphism at p

    Args:
      p: Point to evaluate rank at

    Return:
      Whether or not is diffeomorphism
    """
    return self.is_immersion(p) and self.is_submersion(p)

  def map_is_compatible(self, g: "Map") -> bool:
    """Check if 2 maps are compatable with each other

    Args:
      g: Another map

    Returns:
      boolean
    """
    type_check = isinstance(g, Map)
    domain_check = g.domain == self.domain
    image_check = g.image == self.image
    return type_check and domain_check and image_check

  def __add__(self, g: Union[float,"Map"]) -> "Map":
    """Add two maps together

    Args:
      g: Another map

    Returns:
      sum of maps
    """
    is_map = isinstance(g, Map)
    is_scalar = g in EuclideanSpace(dimension=1)
    assert is_map or is_scalar

    if is_map:
      assert self.map_is_compatible(g)

    def new_f(p, inverse=False):
      if inverse == False:
        if is_map:
          return self.f(p) + g.f(p)
        else:
          return self.f(p) + g
      else:
        assert 0, "Map is not invertible anymore"

    return Map(new_f, domain=self.domain, image=self.image)

  def __radd__(self, g: Union[float,"Map"]) -> "Map":
    """Add g from the right

    Args:
      g: Another map

    Returns:
      sum of maps
    """
    return self + g

  def __mul__(self, a: Union[float,"Map"]) -> "Map":
    """Multiply a map by a scalar or another map

    Args:
      a: A scalar or map

    Returns:
      af
    """
    is_map = isinstance(a, Map)
    is_scalar = a in EuclideanSpace(dimension=1)
    assert is_map or is_scalar

    def new_f(p, inverse=False):
      ap = a(p) if is_map else a
      if inverse == False:
        return ap*self.f(p)
      else:
        return (1/ap)*self.f.inverse(p)

    return Map(new_f, domain=self.domain, image=self.image)

  def __rmul__(self, a: Union[float,"Map"]) -> "Map":
    """Multiply a map by a scalar or another map

    Args:
      a: A scalar or map

    Returns:
      af
    """
    return self*a

  def __sub__(self, g: "Map") -> "Map":
    """Subtract

    Args:
      g: Another map

    Returns:
      f - g
    """
    return self + -1.0*g

  def __neg__(self):
    """Negate this map

    Returns:
      Negative of this map
    """
    return -1.0*self

################################################################################################################

class LinearMap(Map[Input,Output]):
  """A linear map.

  Attributes:
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """

  def determinant(self) -> "Coordinate":
    """Get the determinant for this linear map.  This is
    invariant to the choice of coordinates that are used.

    Args:
      None

    Returns:
      The determinant of this linear map
    """
    assert self.domain.dimension == self.image.dimension
    return jnp.linalg.det(self.get_coordinates())

################################################################################################################

class MultlinearMap(LinearMap[List[Input],Output]):
  """A multilinear map.

  Attributes:
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  pass

class IdentityMap(LinearMap[Input,Output]):
  """The identity map

  Attributes:
  """
  def __init__(self, manifold: Set, **kwargs):
    """Creates a new identity map.

    Args:
      f: A function that maps between Euclidean spaces.
      domain: Must be a set of Real numbers.
      image: Must be a set of Real numbers.
    """
    self.manifold = manifold
    def f(x, inverse=False):
      return x
    super().__init__(lambda x: x, domain=manifold, image=manifold)

class ProjectionMap(Map[Tuple[Point],Point]):
  """Projection map onto the "idx"'th index

  Attributes:
    idx: Index to return
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  def __init__(self, idx, domain: Set, image: Set):
    """Creates a new projection map.

    Args:
      idx: Index to keep
      domain: Domain
      image: Image
    """
    self.idx = idx
    f = lambda x: x[self.idx]
    super().__init__(f, domain=domain, image=image)

NOOP = None

def curry(f: Map, args: Tuple[Union[NOOP,Input]]) -> Map:
  """Like a projection map

  Args:
    f: A function that maps between Euclidean spaces.
    domain: Must be a set of Real numbers.
    image: Must be a set of Real numbers.
  """
  def curried_f(inputs: Tuple[Input]):
    assert len(args) == len(inputs)
    new_args = [inputs if a is None else a for a in args]
    return f(new_args)

  domain = f.domain
  from src.manifold import CartesianProductManifold
  assert isinstance(domain, CartesianProductManifold)
  assert len(args) == len(domain.Ms)
  Ms = []
  for arg, M in zip(args, domain.Ms):
    if arg is None:
      Ms.append(M)

  if len(Ms) > 1:
    new_domain = type(domain)(*Ms)
  else:
    new_domain = Ms[0]
  return type(f)(curried_f, domain=new_domain, image=f.image)

################################################################################################################

class Function(Map[Input,Output]):
  """A map between Euclidean spaces.

  Attributes:
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  def __init__(self, f: Callable[[EuclideanSpace],EuclideanSpace], domain: EuclideanSpace, image: EuclideanSpace):
    """Creates a new function.

    Args:
      f: A function that maps between Euclidean spaces.
      domain: Must be a set of Real numbers.
      image: Must be a set of Real numbers.
    """
    assert isinstance(domain, EuclideanSpace)
    assert isinstance(image, EuclideanSpace)
    super().__init__(f, domain=domain, image=image)

class _InvertibleMixin():
  """A mixin class to give ability to invert a map
  """

  def get_inverse(self):
    """Creates an inverse of the function

    Returns:
      A new Function object that is the inverse of this one.
    """
    def f(x, inverse=False):
      return self.f(x, inverse=not inverse)
    return InvertibleMap(f, domain=self.image, image=self.domain)

  def inverse(self, q: Output) -> Input:
    """Applies the inverse function on q.

    Args:
      q: An input coordinate.

    Returns:
      f^{-1}(q)
    """
    assert q in self.image
    p = self.f(q, inverse=True)
    assert p in self.domain
    return p

class InvertibleFunction(Function[Input,Output], _InvertibleMixin):
  """R <-> R
  """
  pass

class InvertibleMap(Map[Input,Output], _InvertibleMixin):
  """M <-> N
  """
  pass

class Diffeomorphism(Map[Input,Output], _InvertibleMixin):
  """M <-> N
  """
  pass

class InvertibleLinearMap(LinearMap, _InvertibleMixin):
  pass

def compose(*maps: List[Map]) -> Map:
  """Creates a map that is the composition of the provided maps.
  Returns maps[0]*...*maps[-2]*maps[-1]

  Args:
    maps: At least 2 maps.

  Returns:
    Returns maps[0]*...*maps[-2]*maps[-1].
  """
  assert len(maps) >= 2

  # Ensure that all inputs are maps
  for phi in maps:
    assert isinstance(phi, Map)

  # Construct the composition.
  def composition(p, inverse=False):
    def _compose(f, g):
      return lambda *a, **kw: g(f(*a, **kw))
    if inverse == False:
      return reduce(_compose, maps[::-1])(p)
    else:
      inverse_maps = [m.get_inverse() for m in maps]
      return reduce(_compose, inverse_maps)(p)

  # Figure out what return type to use
  map_class = Map
  first_type = type(maps[0])
  if all([type(phi) for phi in maps]):
    map_class = maps[0].__class__

  # Construct the return map
  return map_class(composition, domain=maps[-1].domain, image=maps[0].image)

################################################################################################################
