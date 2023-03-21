from functools import partial
from typing import Callable, List, TypeVar, Generic, Tuple
import src.util
from functools import partial
import jax.numpy as jnp
import jax
from src.set import *
import abc
import src.util as util

__all__ = ["Input",
           "Output",
           "Map",
           "IdentityMap",
           "LinearMap",
           "MultlinearMap",
           "ProjectionMap",
           "Function",
           "InvertibleFunction",
           "InvertibleMap",
           "Diffeomorphism",
           "compose",
           "MatrixColsToTangentBasis",
           "MatrixRowsToCotangentBasis"]

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

  def __call__(self, p: Point) -> Point:
    """Applies the function on p.

    Args:
      p: An input point.

    Returns:
      f(p)
    """
    if util.GLOBAL_CHECK:
      assert p in self.domain
    q = self.f(p)
    if util.GLOBAL_CHECK:
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

class LinearMap(Map[Input,Output]):
  """A linear map.

  Attributes:
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  pass

class MultlinearMap(LinearMap[Input,Output]):
  """A multilinear map.

  Attributes:
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  pass

class IdentityMap(Map[Input,Output]):
  """The identity map

  Attributes:
  """
  def __init__(self, *, manifold: Set):
    """Creates a new identity map.

    Args:
      f: A function that maps between Euclidean spaces.
      domain: Must be a set of Real numbers.
      image: Must be a set of Real numbers.
    """
    self.f = lambda x: x
    self.domain = manifold
    self.image = manifold

  def __call__(self, p: Point) -> Point:
    """Applies the function on p.

    Args:
      p: An input point.

    Returns:
      f(p)
    """
    return p

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

################################################################################################################

class Function(Map[Input,Output]):
  """A map between Euclidean spaces.

  Attributes:
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  def __init__(self, f: Callable[[Reals],Reals], domain: Reals=Reals(), image: Reals=Reals()):
    """Creates a new function.

    Args:
      f: A function that maps between Euclidean spaces.
      domain: Must be a set of Real numbers.
      image: Must be a set of Real numbers.
    """
    assert isinstance(domain, Reals)
    assert isinstance(image, Reals)
    super().__init__(f, domain=domain, image=image)

class _InvertibleMixin():
  """A mixin class to give ability to invert a map
  """

  def get_inverse(self):
    """Creates an inverse of the function

    Returns:
      A new Function object that is the inverse of this one.
    """
    return Map(partial(self.f, inverse=True), domain=self.image, image=self.domain)

  def inverse(self, q: Output) -> Input:
    """Applies the inverse function on q.

    Args:
      q: An input coordinate.

    Returns:
      f^{-1}(q)
    """
    if util.GLOBAL_CHECK:
      assert q in self.image
    p = self.f(q, inverse=True)
    if util.GLOBAL_CHECK:
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

def compose(*maps: List[Map]) -> Map:
  """Creates a map that is the composition of the provided maps.
  Returns maps[K]*...*maps[1]*maps[0]

  Args:
    maps: At least 2 maps.

  Returns:
    Returns maps[K]*...*maps[1]*maps[0].
  """
  assert len(maps) >= 2
  maps = maps[::-1]

  # Ensure that all inputs are maps
  for phi in maps:
    assert isinstance(phi, Map)

  # Ensure that the domains and images are compatible
  image = maps[0].image
  for phi in maps[1:]:
    # assert phi.domain == image
    assert phi.domain, type(image)
    image = phi.image

  # Construct the composition.
  def composition(p, inverse=False):
    if inverse == False:
      def _compose(fs):
        if len(fs) == 1:
          return fs[0].f(p)
        return fs[-1](_compose(fs[:-1]))
      return _compose(maps)

    else:
      def _compose(fs):
        if len(fs) == 1:
          return fs[-1].inverse(p)
        return fs[0].inverse(_compose(fs[1:]))
      return _compose(maps)

  # Figure out what return type to use
  map_class = Map
  first_type = type(maps[0])
  if all([type(phi) for phi in maps]):
    map_class = maps[0].__class__

  # Construct the return map
  return map_class(composition, domain=maps[0].domain, image=maps[-1].image)

################################################################################################################

class MatrixColsToTangentBasis(Map[Matrix,List["TangentVector"]], _InvertibleMixin):
  """Returns the columns of a matrix as a list.

  Attributes:
    TpM: The tangent space that this is a basis of.
  """
  def __init__(self, TpM: "TangentSpace"):
    """Creates a new MatrixToList

    Args:
      TpM: The tangent space that this is a basis of.
    """
    from src.tangent import TangentVector, TangentBasis, TangentBasisSpace
    from src.instances.lie_groups import GeneralLinearGroup
    self.TpM = TpM

    def f(v, inverse=False):
      if inverse == True:
        return jnp.stack([_v.x for _v in v.basis], axis=1)
      else:
        _vx = jnp.split(v, self.TpM.dimension, axis=1) # Split on cols
        return TangentBasis([TangentVector(vx.ravel(), self.TpM) for vx in _vx], self.TpM)

    domain = GeneralLinearGroup(dim=self.TpM.dimension)
    image = TangentBasisSpace(self.TpM)
    super().__init__(f, domain=domain, image=image)

class MatrixRowsToCotangentBasis(Map[Matrix,List["TangentVector"]], _InvertibleMixin):
  """Returns the rows of a matrix as a list.

  Attributes:
    TpM: The tangent space that this is a basis of.
  """
  def __init__(self, coTpM: "CotangentSpace"):
    """Creates a new MatrixToList

    Args:
      coTpM: The cotangent space that this is a basis of.
    """
    assert 0
    from src.cotangent import CotangentVector, CotangentBasisSpace
    from src.instances.lie_groups import GeneralLinearGroup
    self.coTpM = coTpM

    def f(v, inverse=False):
      if inverse == True:
        return jnp.stack([_v.x for _v in v.basis], axis=0)
      else:
        _vx = jnp.split(v, self.coTpM.dimension, axis=0) # Split on rows
        return CotangentBasis([CotangentVector(vx.ravel(), self.coTpM) for vx in _vx], self.coTpM)

    domain = GeneralLinearGroup(dim=self.coTpM.dimension)
    image = CotangentBasisSpace(self.coTpM)
    super().__init__(f, domain=domain, image=image)

################################################################################################################
