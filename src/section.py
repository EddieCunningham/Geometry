from functools import partial
from typing import Callable, List, Optional, Union
import src.util
from functools import partial
from copy import deepcopy
import jax
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
from src.tangent import *
import src.util as util

__all__ = ["Section",
           "VectorField",
           "CovectorField",
           "Frame"]

class Section(Map[Input,Output], abc.ABC):
  """If pi: M -> N is a continuous map, then a section of pi is a continuous right inverse
     for pi, so sigma: N -> M s.t. pi(sigma(.)) = Id(.)

  Attributes:
    pi: The left inverse of sigma.
    f: Function that performs the mapping.
    domain: A set that the input to map lives in.
    image: Where the map goes to.
  """
  def __init__(self, sigma: Map[Input,Output], pi: Map[Output,Input]):
    """Creates a new section.

    Args:
      sigma: Should be a right inverse of pi
      pi: Map that this is a seciton of
    """
    self.sigma = sigma
    self.pi = pi
    self.manifold = self.pi.image
    self.total_space = self.pi.domain
    super().__init__(sigma, domain=self.pi.image, image=self.pi.domain)

  @abc.abstractmethod
  def __call__(self, p: Input) -> Output:
    pass

  def __add__(self, Y: "Section") -> "Section":
    """Add two sections together

    Args:
      Y: Another section

    Returns:
      X + Y
    """
    # Ensure that X and Y are compatable
    assert self.manifold == Y.manifold
    # assert self.total_space == Y.total_space

    class SectionSum(type(self)):
      def __init__(self, X, Y, sigma, pi):
        self.X = X
        self.Y = Y
        Section.__init__(self, sigma, pi)

      def __call__(self, p: Input) -> Output:
        return self.X(p) + self.Y(p)

    return SectionSum(self, Y, self.sigma, self.pi)

  def __radd__(self, Y: "Section") -> "Section":
    return self + Y

  def __rmul__(self, f: Union[Map,float]) -> "Section":
    """Multiply a section with a scalar or function. fX

    Args:
      f: Another map or a scalar

    Returns:
      fX
    """
    is_map = isinstance(f, Map)
    is_scalar = f in Reals(dimension=1)

    assert is_map or is_scalar

    class SectionRHSProduct(type(self)):
      def __init__(self, X, sigma, pi):
        self.X = X
        self.lhs = f
        self.is_float = f in Reals(dimension=1)
        Section.__init__(self, sigma, pi)

      def __call__(self, p: Input) -> Output:
        fp = self.lhs if self.is_float else self.lhs(p)
        return fp*(self.X(p))

    return SectionRHSProduct(self, self.sigma, self.pi)

  def __neg__(self):
    """Negate the vector field

    Returns:
      -X
    """
    return -1.0*self

  def __sub__(self, Y: "Section") -> "Section":
    return self + -Y

################################################################################################################

class VectorField(Section[Point,TangentVector], abc.ABC):
  """A vector field is a smooth section of the projection map from the
  tangent bundle to the base manifold.  Takes a point and outputs
  a tangent vector on the tangent space at the point.  Treat this
  as a section of the tangent bundle, so will need to define s(p) in R^n
  in order to use.

  Attributes:
    X: Function that assigns a tangent vector to every point on the manifold
    M: The manifold that the vector field is defined on
  """
  def __init__(self, M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      M: The base manifold.
    """
    self.manifold = M

    # Construct the bundle
    domain = M
    from src.bundle import TangentBundle
    image = TangentBundle(M)
    sigma = self.__call__
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    super().__init__(sigma, pi)

  @abc.abstractmethod
  def __call__(self, p: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    pass

  def __mul__(self, f: Map) -> Map:
    """Multiply a vector field with a function

    Args:
      f: A map from the manifold to R

    Returns:
      fX
    """
    assert isinstance(f, Map)

    def fun(p: Point):
      return self(p)(f)
    return Map(fun, domain=self.manifold, image=Reals())

################################################################################################################

class CovectorField(Section[Point,CotangentVector], abc.ABC):
  """A covector field is a smooth section of the projection map from the
  cotangent bundle to the base manifold.  Takes a point and outputs
  a tangent vector on the tangent space at the point.  Treat this
  as a section of the tangent bundle, so will need to define s(p) in R^n
  in order to use.

  Attributes:
    X: Function that assigns a tangent vector to every point on the manifold
    M: The manifold that the vector field is defined on
  """
  def __init__(self, M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      M: The base manifold.
    """
    self.manifold = M

    # Construct the bundle
    domain = M
    from src.bundle import CotangentBundle
    image = CotangentBundle(M)
    sigma = self.__call__
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    super().__init__(sigma, pi)

  @abc.abstractmethod
  def __call__(self, p: Point) -> CotangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    pass

################################################################################################################

class Frame(Section[Point,TangentBasis], abc.ABC):
  """A frame is a collection of linearly independent vector fields
  that form a basis for the tangent space.  Treat this as a section
  of the frame bundle, so will need to define s(p) in GL(n,R) in order
  to use.

  Attributes:
    M: Manifold
  """
  def __init__(self, M: Manifold):
    """Creates a new frame

    Args:
      M: Manifold
    """
    self.manifold = M

    domain = M
    from src.bundle import FrameBundle
    image = FrameBundle(M)
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    super().__init__(sigma, pi)

  @abc.abstractmethod
  def __call__(self, p: Point) -> TangentBasis:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    pass

################################################################################################################

class CoFrame(Section[Point,CotangentBasis], abc.ABC):
  """A coframe is a collection of linearly independent covector fields
  that form a basis for the cotangent space.

  Attributes:
    M: Manifold
  """
  def __init__(self, M: Manifold):
    """Creates a new frame

    Args:
      M: Manifold
    """
    self.manifold = M

    domain = M
    from src.bundle import FrameBundle
    image = CoFrameBundle(M)
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    super().__init__(sigma, pi)

  @abc.abstractmethod
  def __call__(self, p: Point) -> CotangentBasis:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    pass
