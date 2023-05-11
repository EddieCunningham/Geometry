from functools import partial
from typing import Callable, List, Optional, Union, Tuple
from collections import namedtuple
import src.util
from functools import partial
import jax
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.tangent import *
from src.cotangent import *
from src.vector import *
from src.section import *
from src.bundle import *
from src.tensor import *
from src.instances.manifolds import EuclideanManifold
import src.util as util
import einops
import itertools
import abc
import math
import copy

__all__ = ["SymmetricTensor",
           "SymmetricTensorSpace",
           "TensorToSymmetricTensor",
           "make_symmetric",
           "symmetric_product",
           "SymmetricTensorBundle",
           "SymmetricTensorField",
           "TensorFieldToSymmetricTensorField",
           "symmetrize_tensor_field",
           "RiemannianMetric",
           "EuclideanMetric",
           "RiemannianManifold",
           "make_riemannian_manifold",
           "TangentToCotangetBundleIsomorphism",
           "CotangentToTangetBundleIsomorphism",
           "gradient_vector_field",
           "gram_schmidt",
           "OrthogonalFrameBundle"]

class SymmetricTensor(Tensor):
  """A symmetric tensor

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    TkTpM: The space of mixed type tensors that this tensor lives on
  """
  def __init__(self, *x: Coordinate, TkTpM: "SymmetricTensorSpace"):
    """Creates a new tensor.

    Args:
      xs: A list of independent coordinates to make tensor
      TkTpM: The tangent space that the tensor lives on.
    """
    assert isinstance(TkTpM, SymmetricTensorSpace)
    self.xs = x
    self.TkTpM = TkTpM
    self.manifold = self.TkTpM.manifold
    self.phi = self.TkTpM.phi
    self.phi_inv = self.phi.get_inverse()
    self.p = self.TkTpM.p
    self.type = self.TkTpM.type

    self.contract = self.TkTpM.get_coordinate_indices()

  def get_dense_coordinates(self) -> Coordinate:
    """Get the coordinates associated with this tensor

    Returns:
      A single array with the coordinates
    """
    # Loop over all permutations of k basis vectors.  Each permutation
    # corresponds to an entry of the coordinate array.
    TpM = TangentSpace(self.p, self.manifold)
    basis = TpM.get_basis()

    coords = jnp.zeros([self.manifold.dimension]*self.type.l)

    for iterate in itertools.product(enumerate(basis), repeat=self.type.l):
      index, Xs = list(zip(*iterate))
      out = self(*Xs)
      coords = coords.at[tuple(index)].set(out)

    return coords

  def __rmul__(self, a: float) -> "SymmetricTensor":
    """Multiply the tensor by a scalar

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    assert a in EuclideanSpace(dimension=1)
    xs = (self.xs[0]*a,) + self.xs[1:]
    return SymmetricTensor(*xs, TkTpM=self.TkTpM)

  def __add__(self, Y: "SymmetricTensor") -> "SymmetricTensor":
    """Add two tensors together.

    Args:
      Y: Another tensor

    Returns:
      Xp + Yp
    """
    # Must be the same type
    if isinstance(Y, CotangentVector):
      Y = as_tensor(Y)
    assert isinstance(Y, SymmetricTensor)
    assert self.type == Y.type
    x_coords = self.get_dense_coordinates()
    y_coords = Y.get_coordinates(self.phi)

    # Need to create a new tensor space to get the correct
    # contraction indices
    TkTpM = SymmetricTensorSpace(self.p, self.type, self.manifold)
    return SymmetricTensor(x_coords + y_coords, TkTpM=TkTpM)

################################################################################################################

class SymmetricTensorSpace(TensorSpace):
  """The space of Symmetric tensors

  Attributes:
    p: The point where the space lives.
    contract: Tells us what types of objects this tensor accepts
    M: The manifold.
  """
  Element = SymmetricTensor

  def __init__(self, p: Point, tensor_type: TensorType, M: Manifold):
    """Creates a new space of mixed tensors on the tangent bundle

    Args:
      p: The point where the space lives.
      type: The tensor type (k,l)
      M: The manifold.
    """
    assert isinstance(tensor_type, TensorType)
    assert tensor_type.k == 0 # Only has covariant tensors
    self.p = p
    self.type = tensor_type
    self.manifold = M

    # Basis elements have indices of increasing value
    import scipy
    self.dimension = scipy.special.comb(self.manifold.dimension + self.type.l - 1, self.type.l, exact=True)

    # Keep track of the chart function for the manifold
    self.phi = self.manifold.get_chart_for_point(self.p)

    VectorSpace.__init__(self, dimension=self.dimension)

  def get_chart_for_point(self, p: Point) -> "Chart":
    """Get a chart to use at point p

    Args:
      The input point

    Returns:
      The chart that contains p in its domain
    """
    assert 0, "Need to implement"

################################################################################################################

class TensorToSymmetricTensor(SymmetricTensor):
  """The symmetrization of a tensor

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    TkTpM: The space of mixed type tensors that this tensor lives on
  """
  def __init__(self, T: Tensor):
    """Creates a symmetrized tensor out of T.  This happens by
    averaging all permutations of the inputs at T.  Only works
    for covariant tensors

    Args:
      T: The tensor
    """
    assert T.type.k == 0
    self.T = T

    # Make a set of dense coordinates
    x = jnp.zeros([self.T.TkTpM.manifold.dimension]*self.T.type.l)

    # Get the new coordinates by summing all of the permutations
    contract = self.T.TkTpM.get_coordinate_indices()
    tokens, reconstruct = util.extract_tokens(contract)

    for token_permutation in itertools.permutations(tokens):
      perm_contract = reconstruct(token_permutation)
      perm_contract += " -> " + " ".join(tokens)
      perm_x = einops.einsum(*self.T.xs, perm_contract)
      x += perm_x

    x /= math.factorial(self.T.type.l)

    # Create a new tensor space here
    sym_TkTpM = SymmetricTensorSpace(self.T.p, self.T.type, self.T.TkTpM.manifold)
    super().__init__(x, TkTpM=sym_TkTpM)

  def call_unvectorized(self, *Xs: List[TangentVector]) -> Coordinate:
    """Apply the tensor to a list of tangent vectors

    Args:
      Xs: A list of tangent vectors

    Returns:
      Value at Xs
    """
    out = 0.0
    for Xs_perm in itertools.permutations(Xs):
      out += self.T(*Xs_perm)

    k_factorial = math.factorial(len(Xs))
    out /= k_factorial
    return out

def make_symmetric(T: Tensor):
  """Symmetrize a tensor.

  Args:
    T: Tensor to symmetrize

  Returns:
    Symmetric version of T
  """
  return TensorToSymmetricTensor(T)

def symmetric_product(a: Tensor, b: Tensor):
  """Symmetric product of 2 tensors

  Args:
    a: Tensor a
    b: Tensor b

  Returns:
    Symmetric product of a and b
  """
  if isinstance(a, CotangentVector):
    a = as_tensor(a)
  if isinstance(b, CotangentVector):
    b = as_tensor(b)

  return make_symmetric(tensor_product(a, b))

################################################################################################################

class SymmetricTensorBundle(TensorBundle):
  """Bundle of Symmetric covariant k-tensor bundle

  Attributes:
    M: The manifold.
  """
  Element = Tensor # The tensor is evaluated at p!

  def __init__(self, tensor_type: TensorType, M: Manifold):
    """Creates a new tensor bundle

    Args:
      type: The tensor type (0,k)
      M: The manifold.
    """
    assert tensor_type.k == 0 # Covariant only

    # The fiber of a tensor bundle is a tensor
    self.manifold = M
    self.type = tensor_type
    import scipy
    self.dimension = scipy.special.comb(self.manifold.dimension + self.type.l - 1, self.type.l, exact=True)
    FiberBundle.__init__(self, M, EuclideanManifold(dimension=self.dimension))

  def contains(self, x: Tensor) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    return x.manifold == self.manifold

  def get_projection_map(self) -> Map[Tensor,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda v: v.p, domain=self, image=self.manifold)

  def _local_trivialization_map(self, inpt: Union[Element,Tuple[Point,"Fiber"]], inverse: bool=False) -> Union[Tuple[Point,"Fiber"],Element]:
    """Contains the actual implementation of the local trivialization.

    Args:
      inpt: Either an element of the fiber bundle or a tuple (point on manifold, fiber)

    Returns:
      Either a tuple (point on manifold, fiber) or an element of the fiber bundle
    """
    if inverse == False:
      return inpt.p, inpt.xs
    else:
      p, xs = inpt
    return SymmetricTensor(*xs, TkTpM=SymmetricTensorSpace(p, self.tensor_type, self.manifold))

  def get_local_trivialization_map(self, T: SymmetricTensor) -> Map[SymmetricTensor,Tuple[Point,Coordinate]]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    assert isinstance(T, SymmetricTensor)

    assert 0, "Need to figure out if this changes for Symmetric tensors"
    product_dimensions = [x.size for x in T.xs]
    Ms = [EuclideanManifold(dimension=dim) for dim in product_dimensions]
    fiber = manifold_carteisian_product(*Ms)

    image = ProductBundle(self.manifold, fiber)
    def Phi(v, inverse=False):
      if inverse == False:
        return v.p, v.xs
      else:
        p, xs = v
        return SymmetricTensor(*xs, TkTpM=SymmetricTensorSpace(p, self.tensor_type, self.manifold))
    return Diffeomorphism(Phi, domain=self, image=image)

class SymmetricTensorField(TensorField, abc.ABC):
  """The symmetrization of a tensor field.  This will have a non symmetric tensor
  field that we will just symmetrize when calling this tensor.

  Attributes:
    type: The type of the tensor field
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, tensor_type: TensorType, M: Manifold):
    """Creates a new tensor field.

    Args:
      type: The tensor type (k,l)
      M: The base manifold.
    """
    assert tensor_type.k == 0
    self.type = tensor_type
    self.manifold = M

    # Construct the bundle
    domain = M
    image = SymmetricTensorBundle(tensor_type, M)
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    Section.__init__(self, pi)

  def __call__(self, *x: Union[Point,VectorField]) -> Union[Map,SymmetricTensor]:
    """Evaluate the tensor field at a point, or evaluate it on vector fields

    Args:
      Xs: Either a point or a list of vector fields

    Returns:
      Tensor at a point, or a map over the manifold
    """
    if isinstance(x[0], VectorField):
      return self.apply_to_co_vector_fields(*x)
    else:
      p = x[0]
      if util.GLOBAL_CHECK:
        assert p in self.manifold
      return self.apply_to_point(p)

  def apply_to_co_vector_fields(self, *Xs: List[VectorField]) -> Map:
    """Evaluate the tensor field on vector fields

    Args:
      Xs: A list of vector fields

    Returns:
      A map over the manifold
    """
    def fun(p: Point):
      return self(p)(*[X(p) for X in Xs])
    return Map(fun, domain=self.manifold, image=EuclideanManifold(dimension=1))

  @abc.abstractmethod
  def apply_to_point(self, p: Point) -> SymmetricTensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    pass

################################################################################################################

class TensorFieldToSymmetricTensorField(SymmetricTensorField, abc.ABC):
  """The symmetrization of a tensor field.  This will have a non Symmetric tensor
  field that we will just alternate when calling this tensor.

  Attributes:
    type: The type of the tensor field
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, T: TensorField, tensor_type: TensorType, M: Manifold):
    """Creates a new tensor field.

    Args:
      type: The tensor type (k,l)
      M: The base manifold.
    """
    self.T = T
    return super().__init__(tensor_type, M)

  def apply_to_co_vector_fields(self, *Xs: List[VectorField]) -> Map:
    """Evaluate the tensor field on vector fields

    Args:
      Xs: A list of vector fields

    Returns:
      A map over the manifold
    """
    # Construct the tensor field
    output_map = None
    for i, Xs_perm in itertools.permutations(Xs):
      term = self.T.apply_to_co_vector_fields(*Xs_perm)
      output_map = term if output_map is None else output_map + term

    k_factorial = 1/math.factorial(len(Xs))
    output_map *= k_factorial
    return output_map

  def apply_to_point(self, p: Point) -> SymmetricTensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    Tp = self.T.apply_to_point(p)
    return make_symmetric(Tp)

def symmetrize_tensor_field(T: TensorField):
  """Alternate a tensor field.

  Args:
    T: Tensor field to alternate

  Returns:
    Symmetric version of T
  """
  return TensorFieldToSymmetricTensorField(T, T.type, T.manifold)

################################################################################################################

class RiemannianMetric(SymmetricTensorField):
  """A riemannian metric is a symmetric covariant 2-tensor field

  Attributes:
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, M: Manifold):
    """Create a new Riemannian metric

    Args:
      M: The base manifold.
    """
    # A riemannian metric is a symmetric covariant 2-tensor field on M
    self.type = TensorType(0, 2)
    self.manifold = M
    super().__init__(self.type, self.manifold)

################################################################################################################

class EuclideanMetric(RiemannianMetric):
  """The Euclidean metric is delta_{ij}dx^i dx^j

  Attributes:
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, dimension: int, coordinate_function: Optional[Callable[[Point,bool],Coordinate]]=None):
    """Create Euclidean space

    Args:
      dimension: Dimension
      coordinate_function: Optionally give a prefered choice of coordinates
    """
    M = EuclideanManifold(dimension, chart=coordinate_function)
    super().__init__(M)

  def apply_to_co_vector_fields(self, *Xs: List[Union[VectorField,CovectorField]]) -> Map:
    """Evaluate the tensor field on (co)vector fields

    Args:
      Xs: A list of (co)vector fields

    Returns:
      A map over the manifold
    """
    def fun(p: Point):
      return self(p)(*[X(p) for X in Xs])
    return Map(fun, domain=self.manifold, image=EuclideanManifold(dimension=1))

  def apply_to_point(self, p: Point) -> Tensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    # The coordinates are delta_{ij}
    coords = jnp.eye(self.manifold.dimension)

    # Construct the tangent covector
    TkTpM = SymmetricTensorSpace(p, self.type, self.manifold)
    return SymmetricTensor(coords, TkTpM=TkTpM)

################################################################################################################

class RiemannianManifold(Manifold):
  """A Riemannian manifold is a smooth manifold with a Riemannian metric

  Attributes:
    g: A Riemannian metric
    atlas: Atlas providing coordinate representation.
  """
  def __init__(self, g: RiemannianMetric, dimension: int=None):
    """Create a Riemannian manifold object.

    Args:
      g: A Riemannian metric
      dimension: Dimension of the manifold
    """
    assert isinstance(g, RiemannianMetric)
    self.g = g
    super().__init__(dimension=dimension)

  def inner_product(self, Xp: TangentVector, Yp: TangentVector) -> Coordinate:
    """Evaluate the inner product between tangent vectors at p.

    Args:
      Xp: A tangent vector
      Yp: A tangent vector

    Returns:
      <Xp,Yp>_g
    """
    gp = self.g(Xp.p)
    return gp(Xp, Yp)

def make_riemannian_manifold(M: Manifold, g: RiemannianMetric) -> RiemannianManifold:
  """Create a Riemannian manifold from an existing manifold and a metric over it.

  Args:
    M: A manifold
    g: A Riemannian metric

  Returns:
    Riemannian manifold
  """
  assert g.manifold == M

  # Create a deep copy of the input manifold instance
  riemannian_manifold = copy.copy(M)

  # Change the class of the copied object to RiemannianManifold
  riemannian_manifold.__class__ = RiemannianManifold

  # Set the Riemannian metric
  riemannian_manifold.g = g

  # Set the atlas
  riemannian_manifold.get_chart_for_point = M.get_chart_for_point

  return riemannian_manifold

################################################################################################################

class TangentToCotangetBundleIsomorphism(InvertibleMap[TangentBundle,CotangentBundle]):
  """Given a Riemannian metric g on M, there is a bundle homomorphism
  from g_hat: TM -> TM* defined by g_hat(v)(w) = g_p(v, w) where
  p in M, v in TpM, g_hat(v) in TpM*

  Attributes:
    g: The Riemannian metric
  """
  def __init__(self, g: RiemannianMetric):
    """Create a bundle isomorphism

    Attributes:
      g: Riemannian metric
    """
    self.g = g

  def apply_to_vector(self, Xp: TangentVector) -> CotangentVector:
    """Converts a tangent vector to a cotangent vector using the Riemannian metric.
    if g = g_{ij}dx^i dx^j, X = X^i d/dx^i, Y = Y^j d/dx^j, then g_hat(X) = g_{ij}X^i dx^j
    This is lowering an index.

    Args:
      Xp: A tangent vector

    Returns:
      g_hat(Xp)
    """
    assert isinstance(Xp, TangentVector)
    p = Xp.p

    # Get the metric at p
    gp = self.g(p)

    # Get the coordinates for g in terms of the coordinate function for Xp
    g_coords = gp.get_coordinates(Xp.phi)

    # Get the coordinates of the new
    g_hatX_coords = jnp.einsum("ij,i->j", g_coords, Xp.x)

    # Form the dual space
    coTpM = Xp.TpM.get_dual_space()

    return CotangentVector(g_hatX_coords, coTpM=coTpM)

  def apply_to_vector_field(self, X: VectorField) -> CovectorField:
    """Turn a vector field into a covector field by lowering an index.

    Args:
      X: A vector field

    Returns:
      g_hat(X)
    """
    class LowerdIndexCovectorField(CovectorField):
      def __init__(self, isomorphism: "TangentToCotangetBundleIsomorphism", X: VectorField):
        self.X = X
        self.isomorphism = isomorphism
        super().__init__(self.X.manifold)

      def apply_to_point(self, p: Point) -> CotangentVector:
        Xp = self.X(p)
        return self.isomorphism.apply_to_vector(Xp)

    return LowerdIndexCovectorField(self, X)

  def __call__(self, v: Union[TangentVector,VectorField]) -> Union[CotangentVector,CovectorField]:
    if isinstance(v, VectorField):
      return self.apply_to_vector_field(v)
    assert isinstance(v, TangentVector)
    return self.apply_to_vector(v)

  def get_inverse(self):
    """Creates an inverse of the function

    Returns:
      A new Function object that is the inverse of this one.
    """
    return CotangentToTangetBundleIsomorphism(self.g)

  def inverse(self, q: Output) -> Input:
    """Applies the inverse function on q.

    Args:
      q: An input coordinate.

    Returns:
      f^{-1}(q)
    """
    return self.get_inverse()(q)

################################################################################################################

class CotangentToTangetBundleIsomorphism(InvertibleMap[TangentBundle,CotangentBundle]):
  """Inverse of TangentToCotangetBundleIsomorphism

  Attributes:
    g: The Riemannian metric
  """
  def __init__(self, g: RiemannianMetric):
    """Create a bundle isomorphism

    Attributes:
      g: Riemannian metric
    """
    self.g = g

  def apply_to_covector(self, wp: CotangentVector) -> TangentVector:
    """Converts a tangent vector to a cotangent vector using the Riemannian metric.
    if g = g_{ij}dx^i dx^j, X = X^i d/dx^i, Y = Y^j d/dx^j, then g_hat(X) = g_{ij}X^i dx^j
    This is lowering an index.

    Args:
      wp: A tangent vector

    Returns:
      g_hat(wp)
    """
    assert isinstance(wp, CotangentVector)
    p = wp.p

    # Get the metric at p
    gp = self.g(p)

    # Get the coordinates for g in terms of the coordinate function for wp
    g_coords = gp.get_coordinates(wp.phi)
    g_inv_coords = jnp.linalg.inv(g_coords)

    # Get the coordinates of the new
    g_inv_hatX_coords = jnp.einsum("ij,j->i", g_inv_coords, wp.x)

    # Form the dual space
    TpM = wp.coTpM.get_dual_space()

    return TangentVector(g_inv_hatX_coords, TpM=TpM)

  def apply_to_covector_field(self, w: CovectorField) -> VectorField:
    """Turn a vector field into a covector field by lowering an index.

    Args:
      w: A vector field

    Returns:
      g_hat(w)
    """
    class RaisedIndexVectorField(VectorField):
      def __init__(self, isomorphism: "CotangentToTangetBundleIsomorphism", w: CovectorField):
        self.w = w
        self.isomorphism = isomorphism
        super().__init__(self.w.manifold)

      def apply_to_point(self, p: Point) -> TangentVector:
        Xp = self.w(p)
        return self.isomorphism.apply_to_covector(Xp)

    return RaisedIndexVectorField(self, w)

  def __call__(self, w: Union[CotangentVector,CovectorField]) -> Union[TangentVector,VectorField]:
    if isinstance(w, CovectorField):
      return self.apply_to_covector_field(w)
    assert isinstance(w, CotangentVector)
    return self.apply_to_covector(w)

  def get_inverse(self):
    """Creates an inverse of the function

    Returns:
      A new Function object that is the inverse of this one.
    """
    return TangentToCotangetBundleIsomorphism(self.g)

  def inverse(self, q: Output) -> Input:
    """Applies the inverse function on q.

    Args:
      q: An input coordinate.

    Returns:
      f^{-1}(q)
    """
    return self.get_inverse()(q)

################################################################################################################

def gradient_vector_field(f: Map[Point,Coordinate], g: RiemannianMetric):
  """Get the gradient vector field of a function using the Riemannian metric g

  Args:
    f: A map from the manifold to the set of reals
    g: A Riemannian metric for the manifold

  Returns:
    grad f
  """
  df = FunctionDifferential(f)
  return CotangentToTangetBundleIsomorphism(g).apply_to_covector_field(df)

################################################################################################################

def gram_schmidt(tangent_basis: TangentBasis, g: RiemannianMetric):
  """Orthogonalize the basis vectors using the Gram-Schmidt algorithm

  Args:
    tangent_basis: A basis for the tangent space
    g: A Riemannian metric for the manifold

  Returns:
    An orthogonalized basis
  """
  TpM = tangent_basis.TpM
  p = TpM.p
  gp = g(p)

  metric_coordinates = gp.get_dense_coordinates()

  # Go to Euclidean coordinates
  L = jnp.linalg.cholesky(metric_coordinates).T # L.T@L = metric_coordinates

  # Get the coordinates of the basis vectors in terms of the metric's coordinate function
  J = jnp.stack([v.get_coordinates(gp.phi) for v in tangent_basis.basis], axis=1)

  # Transform the tangent vector coordinates to Euclidean coordinates
  transformed_J = L@J

  # Do Gram Schmidt
  Q, _ = jnp.linalg.qr(transformed_J)

  # Go back to the coordinates that the metric uses
  new_mat = jnp.linalg.solve(L, Q)

  # Return the new basis
  new_coords = jnp.split(new_mat, TpM.manifold.dimension, axis=1)
  return TangentBasis([TangentVector(x.ravel(), TpM) for x in new_coords], TpM)

################################################################################################################

class OrthogonalFrameBundle(FrameBundle):
  """Frame bundle.  Represents the space of frames that
  we can have over a manifold.

  Attributes:
    M: The manifold.
  """
  Element = TangentBasis # Every element is a basis for the tangent space

  def __init__(self, M: RiemannianManifold):
    """Creates a new frame bundle.

    Args:
      M: The manifold.
    """
    self.g = M.g
    assert isinstance(M, RiemannianManifold)
    from src.instances.lie_groups import OrthogonalGroup
    FiberBundle.__init__(self, M, OrthogonalGroup(dim=M.dimension))

  def contains(self, x: TangentBasis) -> bool:
    """Checks to see if x exists in the bundle.

    Args:
      x: Test point.

    Returns:
      True if p is in the bundle, False otherwise.
    """
    out = True
    for i, Xp in enumerate(x.basis):
      for j, Yp in enumerate(x.basis):
        inner = self.manifold.inner_product(Xp, Yp)
        if util.GLOBAL_CHECK:
          if i == j:
            out = out and jnp.allclose(inner, 1.0)
          else:
            out = out and jnp.allclose(inner, 0.0)
        if out == False:
          import pdb; pdb.set_trace()

    out = x.TpM.manifold == self.manifold
    if out == False:
      import pdb; pdb.set_trace()
    return out
