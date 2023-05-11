from functools import partial
from typing import Callable, List, Optional, Tuple, Generic, Union
import src.util as util
from functools import partial
from copy import deepcopy
import jax.numpy as jnp
import abc
from src.set import *
from src.map import *
from src.manifold import *
from src.tangent import *
from src.cotangent import *
from src.vector_field import *
from src.instances.manifolds import EuclideanManifold
from src.vector import *
from src.instances.lie_groups import GLRn
import diffrax
import abc
import copy

__all__ = ["FiberBundle",
           "ProductBundle",
           "TangentBundle",
           "GlobalDifferential",
           "PrincipalGBundle",
           "CotangentBundle",
           "FrameBundle",
           "CoframeBundle"]

################################################################################################################

class Fiber(Generic[Point]):
  pass

class FiberBundle(Manifold, abc.ABC):
  """A fiber bundle is a space that is locally a product space.  Each point, p, of the manifold
  has an associated fiber, F_p.  The projection map, pi, goes from the total space to the base space.
  All computations will be done using local trivializations.

  Even though Fiber bundles aren't necessarily smooth manifolds, the one's that we'll work with are
  so for convenience we'll treat it as one.

  Attributes:
    F: The fiber F.
    M: The base space.
    E: The total space which looks looks like MxF around points.
  """
  Element = Tuple[Point,Fiber]

  def __init__(self, M: Manifold, F: Fiber):
    """Create a fiber bundle

    Attributes:
      F: The fiber F.
      M: The base space.
    """
    # assert isinstance(M, Manifold)

    self.fiber = F
    self.manifold = M

    total_space_dimension = F.dimension + M.dimension
    super().__init__(dimension=total_space_dimension)

  @abc.abstractmethod
  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space.

    Returns:
      The map x -> p, x in E, p in M
    """
    pass

  def get_subset_for_point(self, p: Point) -> Tuple[Manifold,"FiberBundle"]:
    """Local trivializations are defined over subsets of the base manifold, so
    this function will get get the subset pi^{-1}(U)

    Args:
      p: A point on the base manifold

    Returns:
      A subset of the base manifold and the fiber bundle that projects onto it.
    """
    # Get the nbhd that the local trivialization is defined on
    chart = self.manifold.get_chart_for_point(p)
    U = chart.domain
    restricted_bundle = copy.copy(self)

    # TODO: Need to implement subsets so that we can see if manifolds
    # are compatable.  For example, right now U != self.manifold so some
    # checks will fail like FrameBundle.contains
    # restricted_bundle.manifold = U

    restricted_bundle.is_a_subset = True
    return U, restricted_bundle

  @abc.abstractmethod
  def _local_trivialization_map(self, inpt: Union[Element,Tuple[Point,Fiber]], inverse: bool=False) -> Union[Tuple[Point,Fiber],Element]:
    """Contains the actual implementation of the local trivialization.

    Args:
      inpt: Either an element of the fiber bundle or a tuple (point on manifold, fiber)

    Returns:
      Either a tuple (point on manifold, fiber) or an element of the fiber bundle
    """
    pass

  def get_local_trivialization_map(self, p: Point) -> Map["FiberBundle","ProductBundle"]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    The local trivialization will take as input a point on the fiber bundle
    and output a point on a product bundle.

    lt(x) -> (p, f)

    Args:
      p: A point on the base manifold.  We need this because local trivializations are
      defined from an open nbhd of the manifold.

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    assert p in self.manifold

    # The local trivialization is defined on a subset of the bundle
    U, restricted_bundle = self.get_subset_for_point(p)

    # Get the implementation
    Phi = self._local_trivialization_map

    # This maps to a product bundle
    image = CartesianProductManifold(self.manifold, self.fiber)

    # Return the diffeomorphism
    return Diffeomorphism(Phi, domain=restricted_bundle, image=image)

  def reshape_fiber_trivialization(self, fiber_after_trivialization: Coordinate) -> Coordinate:
    """The fiber part of a trivialization might be a vector or matrix depending
    on whether we're using a vector or frame bundle.

    Args:
      fiber_after_trivialization: The fiber part after a local trivialization
    """
    return fiber_after_trivialization

  def get_chart_for_point(self, x: Element) -> Chart:
    """Get a chart to use at a point x in the total space.  Do this
    by getting a chart of the local trivialization at x.

    Args:
      x: An element of the total space

    Returns:
      The chart that maps x to a real vector
    """
    p = self.get_projection_map()(x)
    lt = self.get_local_trivialization_map(p)
    p, f = lt(x)
    UxF = lt.image
    chart = UxF.get_chart_for_point((p, f))
    return compose(chart, lt)

################################################################################################################

class ProductBundle(CartesianProductManifold, FiberBundle):
  """Trivial bundle is just the product of the base space and fiber.

  Attributes:
    MxF: The cartesian product of M and F
    See ProductManifold for more details
  """
  Element = Tuple[Point,Fiber]

  def __init__(self, M: Manifold, F: Fiber):
    """Creates a new product bundle object.

    Args:
      M: The manifold.
    """
    self.manifold, self.fiber = M, F
    CartesianProductManifold.__init__(self, M, F)
    FiberBundle.__init__(self, M, F)

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return ProjectionMap(0, domain=self, image=self.manifold)

  def _local_trivialization_map(self, inpt: Union[Element,Tuple[Point,Fiber]], inverse: bool=False) -> Union[Tuple[Point,Fiber],Element]:
    """Contains the actual implementation of the local trivialization.

    Args:
      inpt: Either an element of the fiber bundle or a tuple (point on manifold, fiber)

    Returns:
      Either a tuple (point on manifold, fiber) or an element of the fiber bundle
    """
    return inpt

  def contains(self, x: Element) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    p, v = x
    return (p in self.manifold) and (v in self.fiber)

################################################################################################################

class TangentBundle(FiberBundle):
  """Tangent bundle of a manifold.  Represents union of
     tangent spaces at all points.  Elements are tuples
     of the form (p, v) where v is in TpM

  Attributes:
    M: The manifold.
  """
  Element = TangentVector # The tangent vector is evaluated at p!

  def __init__(self, M: Manifold):
    """Creates a new tangent space.

    Args:
      M: The manifold.
    """
    super().__init__(M, EuclideanManifold(dimension=M.dimension))

  def get_projection_map(self) -> Map[TangentVector,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda v: v.p, domain=self, image=self.manifold)

  def _local_trivialization_map(self, inpt: Union[Element,Tuple[Point,Fiber]], inverse: bool=False) -> Union[Tuple[Point,Fiber],Element]:
    """Contains the actual implementation of the local trivialization.

    Args:
      inpt: Either an element of the fiber bundle or a tuple (point on manifold, fiber)

    Returns:
      Either a tuple (point on manifold, fiber) or an element of the fiber bundle
    """
    if inverse == False:
      return inpt.p, inpt.x
    else:
      p, x = inpt
      return TangentVector(x, TangentSpace(p, self.manifold))

  def contains(self, v: TangentVector) -> bool:
    """Checks to see if v exists in the bundle.

    Args:
      v: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    assert isinstance(v, TangentVector)
    return (v.p in self.manifold) and (v.x in self.fiber)

################################################################################################################

class GlobalDifferential(LinearMap[TangentBundle,TangentBundle]):
  """The global differential is the union of all of the
  differentials at every point on the manifold
  """
  def __init__(self, F: Map, M: Manifold):
    """Creates a new global differential
    Args:
      M: The manifold.
    """
    self.F = F
    self.manifold = M
    self.domain = TangentBundle(M)
    self.image = TangentBundle(apply_map_to_manifold(self.F, self.manifold))
    Map.__init__(self, self.__call__, domain=self.domain, image=self.image)

  def __call__(self, x: TangentVector) -> TangentVector:
    """Compute the differential of a tangent vector

    Args:
      x: A tangent vector on the tangent bundle

    Returns:
      The differential of F applied to x
    """
    assert isinstance(x, TangentVector)
    return self.F.get_differential(x.p)(x)

################################################################################################################

class CotangentBundle(TangentBundle):
  """Cotangent bundle of a manifold.  Represents union of
     cotangent spaces at all points.

  Attributes:
    M: The manifold.
  """
  Element = CotangentVector

  def __init__(self, M: Manifold):
    """Creates a new tangent space.

    Args:
      M: The manifold.
    """
    super().__init__(M)

  def _local_trivialization_map(self, inpt: Union[Element,Tuple[Point,Fiber]], inverse: bool=False) -> Union[Tuple[Point,Fiber],Element]:
    """Contains the actual implementation of the local trivialization.

    Args:
      inpt: Either an element of the fiber bundle or a tuple (point on manifold, fiber)

    Returns:
      Either a tuple (point on manifold, fiber) or an element of the fiber bundle
    """
    if inverse == False:
      return inpt.p, inpt.x
    else:
      p, x = inpt
      return CotangentVector(x, CotangentSpace(p, self.manifold))

################################################################################################################

class PrincipalGBundle(FiberBundle, abc.ABC):
  """A principal G-bundle is a fiber bundle where the fiber is a Lie group
  that is also equipped with a differentiable right G-action E x G -> G such
  that the following holds:
  1) E_p = pi^{-1}(p), are the orbits for the G-action (the fibers are orbits)
  2) The local trivialization maps are equivariant, so phi(x*g) = phi(x)*g

  Attributes:
    M: The manifold.
  """

  def __init__(self, M: Manifold, G: "LieGroup"):
    """Create a fiber bundle

    Attributes:
      F: The fiber F.
      M: The base space.
    """
    from src.lie_group import LieGroup
    assert isinstance(G, LieGroup)
    super().__init__(M, G)
    self.G = self.fiber

  @abc.abstractmethod
  def get_action_map(self, *, right: bool) -> Map[Tuple[Fiber,Point],Fiber]:
    """The right action map of G on M.  This is set to
       a translation map by default.

    Args:
      right: Use a right action if true otherwise left action

    Returns:
      theta
    """
    pass

  def fundamental_vector_field(self, A: "LeftInvariantVectorField") -> VectorField:
    """The fundamental vector field corresponding to A is defined at
    u as the tangent vector of the curve ug(t) where g(t) = exp(tA).

    Any vector field over the principal G bundle, P, that can be
    generated like this are called "vertical" tangent vectors.

    Args:
      A: An element of the Lie algebra

    Returns:
      The fundamental vector field
    """
    from src.lie_algebra import LeftInvariantVectorField
    class FundamentalVectorField(VectorField):

      def __init__(self, P: "PrincipalGBundle", A: LeftInvariantVectorField):
        assert isinstance(A, LeftInvariantVectorField)
        assert isinstance(P, PrincipalGBundle)
        self.A = A
        self.P = P
        self.G = self.P.G
        self.right_action = self.P.get_action_map(right=True)
        assert isinstance(self.right_action.domain, CartesianProductManifold)
        assert self.right_action.domain.Ms[0] == self.P
        assert self.right_action.domain.Ms[1] == self.G
        assert self.right_action.image == self.P
        super().__init__(self.P)

      def apply_to_point(self, u: Point) -> TangentVector:
        def _theta_u(g):
          return self.right_action((u, g))
        theta_u = Map(_theta_u, domain=self.G, image=self.P)
        dtheta_u_e = theta_u.get_differential(self.G.e)
        Ae = self.A(self.G.e)
        Astar_u = dtheta_u_e(Ae)
        return Astar_u

    return FundamentalVectorField(self, A)

  def connection_one_form(self, X: VectorField) -> "LeftInvariantVectorField":
    """The connection 1-form turns a fundamental vector field into a left
    invariant vector field.
    w(Astar) = A
    w: χ(P) -> Lie(G)

    If X is a fundamental vector field, then w(X) returns

    Args:
      X: A vector field (not necessarily fundamental vector field)

    Returns:
      Left invariant vector field corresponding to input
    """
    from src.lie_algebra import LeftInvariantVectorField
    class ConnectionOneForm(LieAlgebraValuedDifferentialForm):

      def __init__(self, P: "PrincipalGBundle", X: VectorField):
        assert isinstance(X, VectorField)
        assert isinstance(P, PrincipalGBundle)
        self.X = X
        self.P = P
        self.G = self.P.G
        self.right_action = self.P.get_action_map(right=True)
        assert isinstance(self.right_action.domain, CartesianProductManifold)
        assert self.right_action.domain.Ms[0] == self.P
        assert self.right_action.domain.Ms[1] == self.G
        assert self.right_action.image == self.P
        tensor_type = TensorType(0, 1)
        super().__init__(tensor_type, self.P, self.G.lieG)

      def apply_to_point(self, p: Point) -> LieAlgebraValuedAlternatingTensor:
        pass


    return ConnectionOneForm(self, X)

################################################################################################################

class FrameBundle(PrincipalGBundle):
  """Frame bundle.  Represents the space of frames that
  we can have over a manifold.

  Attributes:
    M: The manifold.
  """
  Element = TangentBasis # Every element is a basis for the tangent space

  def __init__(self, M: Manifold):
    """Creates a new frame bundle.

    Args:
      M: The manifold.
    """
    super().__init__(M, GLRn(dim=M.dimension))

  def contains(self, x: TangentBasis) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    # return True # Should we allow vector fields for the right action map?
    assert isinstance(x, TangentBasis)
    out = x.TpM.manifold == self.manifold
    if out == False:
      import pdb; pdb.set_trace()
    return out

  def get_projection_map(self) -> Map[TangentBasis,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda x: x.TpM.p, domain=self, image=self.manifold)

  def _local_trivialization_map(self, inpt: Union[Element,Tuple[Point,Fiber]], inverse: bool=False) -> Union[Tuple[Point,Fiber],Element]:
    """Contains the actual implementation of the local trivialization.

    Args:
      inpt: Either an element of the fiber bundle or a tuple (point on manifold, fiber)

    Returns:
      Either a tuple (point on manifold, fiber) or an element of the fiber bundle
    """
    if inverse == False:
      tangent_basis = inpt
      p = tangent_basis.TpM.p
      mat = jnp.stack([v.x for v in tangent_basis.basis], axis=1)
      return p, mat
    else:
      p, mat = inpt
      xs = jnp.split(mat, self.manifold.dimension, axis=1) # Split on cols

      # Need to recreate this so that we can pass gradients through p!
      TpM = TangentSpace(p, self.manifold)
      basis = TangentBasis([TangentVector(x.ravel(), TpM) for x in xs], TpM)
      return basis

  def reshape_fiber_trivialization(self, fiber_after_trivialization: Coordinate):
    dim = self.manifold.dimension
    return fiber_after_trivialization.reshape((dim, dim))

  def get_action_map(self, *, right: bool) -> Map[Tuple[TangentBasis, Point], TangentBasis]:
    """The right action map of G on M.  This is set to
       a translation map by default.

    Args:
      right: Use a right action if true otherwise left action

    Returns:
      theta
    """
    assert isinstance(right, bool)

    def theta(Fg: Tuple[TangentBasis, InvertibleMatrix]) -> TangentBasis:

      if right:
        Fp, g = Fg
      else:
        g, Fp = Fg

      assert isinstance(Fp, TangentBasis)
      assert isinstance(g, jnp.ndarray)

      p = Fp.TpM.p

      # Get a local trivialization at p
      lt = self.get_local_trivialization_map(p)

      # Turn the basis into a matrix
      p, basis_as_matrix = lt(Fp)

      # Apply the transformation
      if right:
        new_basis_as_matrix = basis_as_matrix@g
      else:
        new_basis_as_matrix = g@basis_as_matrix

      # Get back a basis
      new_basis = lt.inverse((p, new_basis_as_matrix))

      return new_basis

    if right:
      domain = CartesianProductManifold(self, GLRn(dim=self.manifold.dimension))
    else:
      domain = CartesianProductManifold(GLRn(dim=self.manifold.dimension), self)
    return Map(theta, domain=domain, image=self)

################################################################################################################

class CoframeBundle(FrameBundle):
  """Co-frame bundle.  Represents the space of coframes that
  we can have over a manifold.

  Attributes:
    M: The manifold.
  """
  Element = CotangentBasis

  def contains(self, x: CotangentBasis) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    return x.coTpM.manifold == self.manifold

  def get_projection_map(self) -> Map[Element,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda x: x.coTpM.p, domain=self, image=self.manifold)

  def _local_trivialization_map(self, inpt: Union[Element,Tuple[Point,Fiber]], inverse: bool=False) -> Union[Tuple[Point,Fiber],Element]:
    """Contains the actual implementation of the local trivialization.

    Args:
      inpt: Either an element of the fiber bundle or a tuple (point on manifold, fiber)

    Returns:
      Either a tuple (point on manifold, fiber) or an element of the fiber bundle
    """
    if inverse == False:
      cotangent_basis = inpt
      p = cotangent_basis.coTpM.p
      mat = jnp.stack([v.x for v in cotangent_basis.basis], axis=0)
      return p, mat
    else:
      p, mat = inpt
      xs = jnp.split(mat, self.manifold.dimension, axis=0) # Split on rows

      # Need to recreate this so that we can pass gradients through p!
      coTpM = CotangentSpace(p, self.manifold)
      basis = CotangentBasis([CotangentVector(x.ravel(), coTpM) for x in xs], coTpM)
      return basis

################################################################################################################

# class OrbitSpace(Manifold):

#   def __init__(self, M: Manifold, G: LieGroup):
#     self.M = M
#     self.G = G
#     dimension = self.M.dimension - self.G.dimension
#     super().__init__(dimension=dimension)

#   def get_atlas(self) -> Atlas:
#     """Construct the atlas for the orbit space.  Do this by first
#     finding a chart for M and then adapting it to the G-action, which
#     we'll do by ensuring that the tangent space is aligned with

#     Args:
#       atlas: Atlas providing coordinate representation.
#     """
#     new_charts = []
#     for chart in self.M.atlas.charts:

#       rotation_matrix = None

#       def phi(x, inverse=False):
#         if inverse == False:
#           # Get a chart from M
#           phi = self.M.get_chart_for_point(x)

#           # Get the orbit map
#           theta_p = self.G.get_orbit_map(x, self.manifold)

#           # Get the differential of the orbit map at the identity element
#           dtheta_p_e = theta_p.get_differential(self.G.e)

#           # Get a basis for the Lie algebra
#           basis = G.lieG.TeG.get_basis()

#           # Each basis vector for TeG will be pushed through the orbit map
#           # to get a basis vector for the tangent space of the orbit.
#           # Then we'll keep track of the coordinates of the tangent vectors
#           # in terms of the chart for phi
#           coords = []
#           for i, ei in enumerate(basis):
#             dtheta_ei = dtheta_p_e(ei)
#             coord = dtheta_ei.get_coordinates(phi)
#             coords.append(coord)

#           coords = jnp.array(coords)

#           # The coordinates should form an NxK matrix called B.
#           # We need to find a matrix A s.t. AB = [C \\ 0]
#           # TODO: Finish this?
#           nonlocal rotation_matrix

#           import pdb; pdb.set_trace()
#         else:
#           # We need to keep track of the information we throw away in the
#           # forward pass so that we can invert correctly
#           assert 0

#       new_chart = Chart(phi, domain=self, image=EuclideanSpace(dimension=self.dimension))
#       new_charts.append(new_chart)

#     return Atlas(new_charts)

# ################################################################################################################
