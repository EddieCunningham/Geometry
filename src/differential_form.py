from functools import partial
from typing import Callable, List, Optional, Union, Tuple
from collections import namedtuple
import src.util
from functools import partial, reduce
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
import copy
import math

__all__ = ["Permutation",
           "PermutationSet",
           "MultiIndex",
           "AlternatingTensor",
           "ElementaryAlternatingTensor",
           "kronocker_delta",
           "AlternatingTensorSpace",
           "make_alternating",
           "wedge_product",
           "InteriorProductAlternatingTensor",
           "interior_product",
           "AlternatingTensorBundle",
           "DifferentialForm",
           "TensorFieldToDifferentialForm",
           "make_alternating_tensor_field",
           "WedgeProductForm",
           "wedge_product_form",
           "InteriorProductForm",
           "interior_product_form",
           "exterior_derivative",
           "ExteriorDerivativeForm",
           "PullbackDifferentialForm",
           "pullback_differential_form"]

################################################################################################################

class Permutation(Map[List[int],List[int]]):
  """A permutation will rearragnge elements of a list.

  Attributes:
    indices: The order to rearrange the items.
  """
  def __init__(self, indices: List[int]):
    """Create a new Permuation object

    Args:
      indices: The order to rearrange the items.
    """
    self.indices = indices
    self.k = len(self.indices)

  def __call__(self, elements: List[int]):
    """Rearrange elements to have this permutation's order.

    Args:
      elements: A list of objects

    Returns:
      Rearranged list of elements
    """
    assert len(elements) == self.k
    return [elements[i] for i in self.indices]

  def get_parity(self):
    """Get the parity of this permutation

    Returns:
      Parity of permutation
    """
    import sympy.combinatorics.permutations as sympy_perm
    return sympy_perm.Permutation(self.indices).parity()

class PermutationSet(Set):
  """The set of permutations of k elements

  Attributes:
    k: The number of elements that we'll permute
  """
  def __init__(self, k: int):
    """Create a new Permuation object

    Args:
      k: The number of elements to permute over
    """
    self.k = k

  def contains(self, p: Point) -> bool:
    """Checks to see if p exists in this set.

    Args:
      p: Test point.

    Returns:
      True if p is in the set, False otherwise.
    """
    return isinstance(p, Permutation) and (p.k == self.k)

  def get_signed_permutations(self):
    """Algorithm that yields the permutations of elements
    by swapping different elements of the list.  GPT4 generated
    most of this code (it seems to base it off of the pseudo
    code from the wikipedia page of heap's algorithm).

    Returns:
      An iterator for the permutations
    """
    # Get the original indices
    indices = list(range(self.k))
    elements = list(indices)

    n = len(elements)
    c = [0]*n
    parity = 1
    yield parity, Permutation(elements)

    i = 0
    while i < n:
      if c[i] < i:
        if i%2 == 0:
          elements[0], elements[i] = elements[i], elements[0]
        else:
          elements[c[i]], elements[i] = elements[i], elements[c[i]]
        parity *= -1
        yield parity, Permutation(elements)
        c[i] += 1
        i = 0
      else:
        c[i] = 0
        i += 1

  @staticmethod
  def signed_permutations_of_list(elements: List):
    """Static version of get_signed_permutations to apply directly
    to a list of elements.

    Args:
      elements: A list of objects

    Returns:
      An iterator for the permutations
    """
    k = len(elements)
    pset = PermutationSet(k)
    for parity, perm in pset.get_signed_permutations():
      yield parity, perm(elements)

################################################################################################################

class MultiIndex(list):
  """A multi-index of length k is an ordered k-tuple of positive integers

  Attributes:
    indices: A list of positive integers
  """
  def __init__(self, indices: List[int]):
    """Create a new multi-index object

    Args:
      index: Which indices vector to get
    """
    assert all([isinstance(i, int) for i in indices])
    super().__init__(indices)

  def is_increasing(self) -> bool:
    """Check if this multi index is increasing

    Returns:
      True if is increasing otherwise false
    """
    pairs = zip(self, self[1:])
    return all([i < j for i, j in pairs])

################################################################################################################

class AlternatingTensor(Tensor):
  """The alternatization of a tensor

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    TkTpM: The space of mixed type tensors that this tensor lives on
  """
  def __init__(self, *x: Coordinate, TkTpM: "AlternatingTensorSpace"):
    """Creates a new tensor.

    Args:
      xs: A list of independent coordinates to make tensor
      TkTpM: The tangent space that the tensor lives on.
    """
    assert isinstance(TkTpM, AlternatingTensorSpace)
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

    unique_coords = jnp.zeros([self.manifold.dimension]*self.type.l)
    coords = jnp.zeros_like(unique_coords)

    # The only unique values come from unique combinations of basis vectors
    for iterate in itertools.combinations(enumerate(basis), self.type.l):
      index, Xs = list(zip(*iterate))
      out = self(*Xs)

      # Update the unique elements of the coordinate array
      unique_coords = unique_coords.at[index].set(out)

    # The remaining elements are filled in by permuting the unique elements
    for parity, perm_index in PermutationSet.signed_permutations_of_list(list(range(len(coords.shape)))):
      coords += parity*unique_coords.transpose(perm_index)

    return coords

  def __rmul__(self, a: float) -> "AlternatingTensor":
    """Multiply the tensor by a scalar

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    assert a in EuclideanSpace(dimension=1)
    xs = (self.xs[0]*a,) + self.xs[1:]
    return AlternatingTensor(*xs, TkTpM=self.TkTpM)

  def __add__(self, Y: "AlternatingTensor") -> "AlternatingTensor":
    """Add two tensors together.

    Args:
      Y: Another tensor

    Returns:
      Xp + Yp
    """
    # Must be the same type
    if isinstance(Y, CotangentVector):
      Y = as_tensor(Y)
    assert isinstance(Y, AlternatingTensor)
    assert self.type == Y.type
    x_coords = self.get_dense_coordinates()
    y_coords = Y.get_coordinates(self.phi)

    # Need to create a new tensor space to get the correct
    # contraction indices
    TkTpM = AlternatingTensorSpace(self.p, self.type, self.manifold)
    return AlternatingTensor(x_coords + y_coords, TkTpM=TkTpM)

  def get_basis_elements(self) -> List[Tuple[Coordinate,"ElementaryAlternatingTensor"]]:
    """Decompose this alternating tensor into its basis elements.

    Returns:
      A list of (scalar, basis element)
    """
    TpM = TangentSpace(self.p, self.manifold)
    basis = TpM.get_basis()

    out = []

    # The only unique values come from unique combinations of basis vectors
    for iterate in itertools.combinations(enumerate(basis), self.type.l):
      index, Xs = list(zip(*iterate))

      # Get the coordinate
      wI = self(*Xs)

      # Form the basis element
      eI = ElementaryAlternatingTensor(MultiIndex(index), TkTpM=self.TkTpM)

      out.append((wI, eI))

    return out

################################################################################################################

class ElementaryAlternatingTensor(AlternatingTensor):
  """An elementary alternating tensor can form a basis for the space of alternating tensors.

  Attributes:
    I: A multiindex object
  """
  def __init__(self, I: MultiIndex, TkTpM: "AlternatingTensorSpace"):
    """Create a new elementary tensor

    Args:
      I: A multiindex that sort of represents an index in a multidimensional array.
      TkTpM: The tensor space
    """
    self.I = I
    self.k = len(self.I)
    self.TkTpM = TkTpM
    assert TkTpM.type.k == 0 # Must be covariant
    assert TkTpM.type.l == self.k
    self.manifold = self.TkTpM.manifold
    x = self.get_dense_coordinates() # TODO: See when we can delay computing this.
    super().__init__(x, TkTpM=TkTpM)

  def get_dense_coordinates(self) -> Coordinate:
    """Get the coordinates associated with this elementary tensor.
    Don't need to bother actually constructing the basis vectors here.

    Returns:
      A single array with the coordinates
    """
    # Loop over all permutations of k basis vectors.  Each permutation
    # corresponds to an entry of the coordinate array.
    coords = jnp.zeros([self.manifold.dimension]*self.k)
    for index in itertools.permutations(range(self.manifold.dimension), self.k):
      J = MultiIndex(index)
      out = kronocker_delta(self.I, J)
      coords = coords.at[tuple(J)].add(out)
    return coords

  def __call__(self, *Xs: List[TangentVector]) -> Coordinate:
    """Apply the tensor to the inputs.

    Args:
      Xs: A list of tangent vectors

    Returns:
      Determinant of rows of coordinates
    """
    assert all([isinstance(X, TangentVector) for X in Xs])
    assert len(Xs) == len(self.I)

    # Get the cooridnates for the tangent vectors in terms of the basis from
    # this manifold
    coords = [X.get_coordinates(self.phi) for X in Xs]

    # Put these together in a matrix
    coord_matrix = jnp.stack(coords, axis=1)

    # Get the rows of each coordinate
    sub_matrix = coord_matrix[self.I,:]

    return jnp.linalg.det(sub_matrix)

  def call_unvectorized(self, *Xs: List[TangentVector]) -> Coordinate:
    """Apply the tensor to the inputs.

    Args:
      Xs: A list of tangent vectors

    Returns:
      Determinant of rows of coordinates
    """
    assert all([isinstance(X, TangentVector) for X in Xs])
    assert len(Xs) == len(self.I)

    # Construct a basis for the tangent space
    basis = CotangentSpace(self.p, self.manifold).get_basis()

    # Reorder the basis vectors according to the multi-index
    reordered_basis = [basis[i] for i in self.I]

    # Apply the basis to each tangent vector
    matrix = jnp.zeros((self.k, self.k))
    for i, e in enumerate(reordered_basis):
      for j, v in enumerate(Xs):
        matrix = matrix.at[i,j].set(e(v))

    return jnp.linalg.det(matrix)

def kronocker_delta(I: MultiIndex, J: MultiIndex):
  # If I and J have any repeated index, then value is 0
  if len(set(I)) != len(I):
    return 0

  if len(set(J)) != len(J):
    return 0

  # If there is no permutation to get from I to J, return 0
  if set(I) != set(J):
    return 0

  # Otherwise, return the parity of the permutation to go from I to J
  # TODO: code up more efficient method
  pset = PermutationSet(len(I))
  for s, perm in pset.get_signed_permutations():
    if perm(I) == J:
      return s

  assert 0, "Can't get here"

################################################################################################################

class AlternatingTensorSpace(TensorSpace):
  """The space of alternating tensors

  Attributes:
    p: The point where the space lives.
    contract: Tells us what types of objects this tensor accepts
    M: The manifold.
  """
  Element = AlternatingTensor

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
    self.dimension = scipy.special.comb(self.manifold.dimension, self.type.l, exact=True)

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

  def get_basis(self) -> List["ElementaryAlternatingTensor"]:
    """Get a basis of alternating tensors for this space

    Returns:
      A list of tensors that form a basis for the tensor space
    """
    out = []
    for index in itertools.combinations(range(self.manifold.dimension), self.type.l):
      assert index == sorted(index)
      import pdb; pdb.set_trace()
      eI = ElementaryAlternatingTensor(MultiIndex(index), TkTpM=self)
      out.append(eI)
    return out

################################################################################################################

class TensorToAlternatingTensor(AlternatingTensor):
  """The alternatization of a tensor

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    TkTpM: The space of mixed type tensors that this tensor lives on
  """
  def __init__(self, T: Tensor):
    """Creates a alternating tensor out of T.  This happens by
    averaging all signed permutations of the inputs at T.  Only works
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

    for i, (perm_sign, token_permutation) in enumerate(PermutationSet.signed_permutations_of_list(tokens)):
      perm_contract = reconstruct(token_permutation)
      perm_contract += " -> " + " ".join(tokens)
      perm_x = perm_sign*einops.einsum(*self.T.xs, perm_contract)
      x += perm_x

    x /= math.factorial(self.T.type.l)

    # Create a new tensor space here
    sym_TkTpM = AlternatingTensorSpace(self.T.p, self.T.type, self.T.TkTpM.manifold)
    super().__init__(x, TkTpM=sym_TkTpM)

  def call_unvectorized(self, *Xs: List[TangentVector]) -> Coordinate:
    """Apply the tensor to a list of tangent vectors

    Args:
      Xs: A list of tangent vectors

    Returns:
      Value at Xs
    """
    out = 0.0
    for i, (perm_sign, Xs_perm) in enumerate(PermutationSet.signed_permutations_of_list(Xs)):
      out += perm_sign*self.T(*Xs_perm)

    k_factorial = math.factorial(len(Xs))
    out /= k_factorial
    return out

################################################################################################################

def make_alternating(T: Tensor):
  """make_alternating a tensor.

  Args:
    T: Tensor to make alternating

  Returns:
    Alternating version of T
  """
  return TensorToAlternatingTensor(T)

################################################################################################################

def wedge_product(*Ts: List[AlternatingTensor]) -> AlternatingTensor:
  """Alternating product of alternating tensors

  Args:
    Ts: A list of alternating tensors

  Returns:
    Alternating product of the input tensors
  """
  Ts = [as_tensor(T) for T in Ts]

  num = math.factorial(sum([T.type.l for T in Ts]))
  den = reduce(lambda x, y: x*y, [math.factorial(T.type.l) for T in Ts])

  const = num/den
  return const*make_alternating(tensor_product(*Ts))

################################################################################################################

class InteriorProductAlternatingTensor(AlternatingTensor):
  """An alternating tensor that comes from interior multiplication

  Attributes:
    w: The base alternating tensor
    v: The vector that we put in the first slot of w
  """
  def __init__(self, w: AlternatingTensor, v: TangentVector):
    """Create a new interior multiplication tensor

    Args:
    w: The base alternating tensor
    v: The vector that we put in the first slot of w
    """
    assert isinstance(w, AlternatingTensor)
    assert isinstance(v, TangentVector)
    self.w = w
    self.v = v
    self.p = self.w.p
    self.manifold = self.w.manifold
    self.type = TensorType(0, self.w.type.l - 1)
    TkTpM = AlternatingTensorSpace(self.w.p, self.type, self.w.manifold)
    x = self.get_dense_coordinates()
    super().__init__(x, TkTpM=TkTpM)

  def __call__(self, *Xs: List[TangentVector]) -> Coordinate:
    """Apply the tensor to the inputs.

    Args:
      Xs: A list of tangent vectors

    Returns:
      Determinant of rows of coordinates
    """
    assert all([isinstance(X, TangentVector) for X in Xs])
    assert len(Xs) == self.w.type.l - 1
    return self.w(self.v, *Xs)

def interior_product(w: AlternatingTensor, v: TangentVector):
  """Given an alternating covariant k-tensor and vector v, the
  interior product, i_v(w) returns an alternating tensor of degree k-1
  that puts v in the first slot of w like w(v,-,...,-)

  Args:
    w: An alternating covariant tensor
    v: A vector

  Returns:
    The interior product i_v(w)
  """
  if w.type.l == 1:
    # Return a point
    return w(v)
  return InteriorProductAlternatingTensor(w, v)

################################################################################################################

class AlternatingTensorBundle(TensorBundle):
  """Bundle of alternating covariant k-tensor bundle

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
    self.dimension = scipy.special.comb(self.manifold.dimension, self.type.l, exact=True)
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
    return AlternatingTensor(*xs, TkTpM=AlternatingTensorSpace(p, self.tensor_type, self.manifold))

  def get_local_trivialization_map(self, T: AlternatingTensor) -> Map[AlternatingTensor,Tuple[Point,Coordinate]]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    assert isinstance(T, AlternatingTensor)

    assert 0, "Need to figure out if this changes for alternating tensors"
    product_dimensions = [x.size for x in T.xs]
    Ms = [EuclideanManifold(dimension=dim) for dim in product_dimensions]
    fiber = manifold_carteisian_product(*Ms)

    image = ProductBundle(self.manifold, fiber)
    def Phi(v, inverse=False):
      if inverse == False:
        return v.p, v.xs
      else:
        p, xs = v
        return AlternatingTensor(*xs, TkTpM=AlternatingTensorSpace(p, self.tensor_type, self.manifold))
    return Diffeomorphism(Phi, domain=self, image=image)

################################################################################################################

class DifferentialForm(TensorField, abc.ABC):
  """A differential k form

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
    image = AlternatingTensorBundle(tensor_type, M)
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    Section.__init__(self, pi)

  def __call__(self, *x: Union[Point,VectorField]) -> Union[Map,AlternatingTensor]:
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
  def apply_to_point(self, p: Point) -> AlternatingTensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    pass

  def get_coordinate_functions_with_basis(self) -> List[Tuple[Map[Manifold,Coordinate],"ElementaryDifferentialForm"]]:
    """Get the coordinate functions associated with basis elements

    Returns:
      A list of coordinate functions
    """
    assert self.manifold.dimension >= self.type.l

    # Get the coordinate frame over the manifold
    coordinate_frame = StandardCoordinateFrame(self.manifold)
    basis_vector_fields = coordinate_frame.to_vector_field_list()

    dual_frame = coordinate_frame.get_dual_coframe()
    basis_covector_fields = dual_frame.to_covector_field_list()

    out = []

    # The only unique values come from unique combinations of basis vectors
    for iterate in itertools.combinations(enumerate(zip(basis_vector_fields, basis_covector_fields)), self.type.l):
      index, Ee = zip(*iterate)
      E, e = zip(*Ee)

      # Get the coordinate
      wI = self(*E)

      # Get the corresponding basis element
      eI = wedge_product_form(*e)
      eI.I = MultiIndex(index) # TODO: MAKE BASIS FIELD
      out.append((wI, eI))

    assert len(out) > 0
    return out

################################################################################################################

class TensorFieldToDifferentialForm(DifferentialForm, abc.ABC):
  """The symmetrization of a tensor field.  This will have a non alternating tensor
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
    for i, (perm_sign, Xs_perm) in enumerate(PermutationSet.signed_permutations_of_list(Xs)):
      term = float(perm_sign)*self.T.apply_to_co_vector_fields(*Xs_perm)
      output_map = term if output_map is None else output_map + term

    k_factorial = 1/math.factorial(len(Xs))
    output_map *= k_factorial
    return output_map

  def apply_to_point(self, p: Point) -> AlternatingTensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    Tp = self.T.apply_to_point(p)
    return make_alternating(Tp)

def make_alternating_tensor_field(T: TensorField):
  """Alternate a tensor field.

  Args:
    T: Tensor field to alternate

  Returns:
    Alternating version of T
  """
  return TensorFieldToDifferentialForm(T, T.type, T.manifold)

################################################################################################################

class WedgeProductForm(DifferentialForm):
  """A differential form coming from a wedge product of differential forms

  Attribues:
    ws: A list of differential forms
  """
  def __init__(self, *ws: List[DifferentialForm]):
    self.manifold = ws[0].manifold
    assert all(w.manifold == self.manifold for w in ws)
    self.ws = ws
    self.type = reduce(lambda x, y: x + y, [w.type for w in self.ws])
    super().__init__(self.type, self.manifold)

  def apply_to_point(self, p: Point) -> AlternatingTensor:
    """Evaluate the form at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    return wedge_product(*[w(p) for w in self.ws])

def wedge_product_form(*ws: List[DifferentialForm]) -> DifferentialForm:
  return WedgeProductForm(*ws)

################################################################################################################

class InteriorProductForm(DifferentialForm):
  """A differential form coming from an interior product

  Attribues:
    w: The base differential form
    v: The vector field that we put in the first slot of w
  """
  def __init__(self, w: DifferentialForm, X: VectorField):
    """Creates a new interior product differential form

    Args:
      w: The base differential form
      X: A vector field to fill the first slot of w
    """
    assert isinstance(w, DifferentialForm)
    assert isinstance(X, VectorField)
    self.w = w
    self.X = X
    self.manifold = self.w.manifold
    self.type = TensorType(0, self.w.type.l - 1)
    super().__init__(self.type, self.manifold)

  def apply_to_co_vector_fields(self, *Xs: List[VectorField]) -> Map:
    """Evaluate the form on (co)vector fields

    Args:
      Xs: A list of (co)vector fields

    Returns:
      A map over the manifold
    """
    assert len(Xs) == self.w.type.l - 1
    return self.w.apply_to_co_vector_fields(self.X, *Xs)

  def apply_to_point(self, p: Point) -> AlternatingTensor:
    """Evaluate the form at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    return interior_product(self.w(p), self.X(p))

def interior_product_form(w: DifferentialForm, X: VectorField) -> DifferentialForm:
  """Create an interior product form from w and X

  Args:
    w: The base differential form
    X: A vector field to fill the first slot of w

  Returns:
    InteriorProductForm
  """
  if w.type.l == 1:
    # Return a function
    return w(X)
  return InteriorProductForm(w, X)

################################################################################################################

class ExteriorDerivativeForm(DifferentialForm):
  """The exterior derivative of a differential form
  """
  def __init__(self, w: DifferentialForm):
    """Creates a new exterior derivative differential form

    Args:
      w: The base differential form
    """
    if isinstance(w, CovectorField):
      w = as_tensor_field(w)

    assert isinstance(w, DifferentialForm)
    self.w = w
    self.manifold = self.w.manifold
    assert self.w.type.k == 0
    self.type = TensorType(self.w.type.k, self.w.type.l + 1)
    super().__init__(self.type, self.manifold)

  def apply_to_co_vector_fields(self, *Xs: List[VectorField]) -> Map:
    """Evaluate this form on vector fields.  We'll try to parallelize
    as much as we can by using the invariant formula for the exterior derivative
    but theres not much that we can do to get around the fact that we need to
    compute a lot of lie brackets and apply the vector field to a lot of functions.

    Args:
      Xs: A list of vector fields

    Returns:
      A map over the manifold
    """
    assert len(Xs) == self.type.l

    # First, compute all of the form terms that we'll need
    form_terms = []
    for i in range(len(Xs)):
      form_terms.append(self.w(*(Xs[:i] + Xs[i+1:])))

    # Next, compute all of the lie bracket terms that we'll need
    from src.lie_derivative import lie_bracket
    lie_bracket_terms = []
    for i, Xi in enumerate(Xs):
      for j, Xj in enumerate(Xs):
        if i < j:
          sign = (-1.0)**(i + j)
          lie_bracket_terms.append(sign*lie_bracket(Xi, Xj))

    def fun(p: Point):
      # Compute the values of each of the tangent vectors
      Xs_p = [X(p) for X in Xs]

      # Compute the values of the form terms.  This is the first sum in the invariant formula
      out = 0.0
      for i, (Xp, wXs) in enumerate(zip(Xs_p, form_terms)):
        sign = (-1)**i
        out += sign*Xp(wXs)

      # The only thing that we can vmap is applying the tangent vectors to the form
      wp = self.w(p)

      # Compute the lie bracket values
      lie_bracket_vectors = [lie_bracket(p) for lie_bracket in lie_bracket_terms]

      # Extract all of the coordinates from the tangent vectors
      # IMPORTANT: THIS ASSUMES THAT EVERYTHING USES THE SAME CHART
      # TODO: Check that everything uses the same chart function?
      Xs_coords = [Xp.x for Xp in Xs_p]

      # Compute the values of the lie brackets
      lb_coords = [lb.x for lb in lie_bracket_vectors]
      lb_coords_iter = iter(lb_coords)

      # Arrange all of the coordinates into a matrix so that we can vmap
      all_coords = []
      for i in range(len(Xs)):
        for j in range(len(Xs)):
          if i < j:
            bracket_coords = next(lb_coords_iter)
            coords = [bracket_coords] + Xs_coords[:i] + Xs_coords[i+1:j] + Xs_coords[j+1:]
            all_coords.append(coords)
      all_coords = jnp.array(all_coords)

      # Create the tangent space
      TpM = TangentSpace(p, self.manifold)

      def apply_w(coords):
        # Create the tangent vectors for the coordinates
        vectors = [TangentVector(x, TpM) for x in coords]
        return wp(*vectors)

      # vmap over all of the terms in the second sum of the invariant
      # formula for the exterior derivative
      lb_terms = jax.vmap(apply_w)(all_coords)

      # Aggregate the terms
      total_out = out + lb_terms.sum()
      return total_out

    return Map(fun, domain=self.manifold, image=EuclideanManifold(dimension=1))

  def apply_to_point(self, p: Point) -> AlternatingTensor:
    """We'll apply this form to points by wrapping the apply to vector fields function.

    Args:
      p: A point on the manifold

    Returns:
      A dummy alternating tensor
    """
    class ExteriorDerivativeTensor(AlternatingTensor):
      def __init__(self, wrapper_class, TkTpM: AlternatingTensorSpace):
        self.wrapper_class = wrapper_class
        super().__init__(None, TkTpM=TkTpM)
        self.xs = (self.get_dense_coordinates(),)

      def __call__(self, *Xs: List[TangentVector]) -> Coordinate:
        # Apply the code above
        class TrivialVF(VectorField):
          def __init__(self, v):
            self.tangent_vector = v
            super().__init__(v.TpM.manifold)

          def apply_to_point(self, p: Point) -> TangentVector:
            return self.tangent_vector

        Xs_vf = [TrivialVF(X) for X in Xs]
        return self.wrapper_class.apply_to_co_vector_fields(*Xs_vf)(p)

    TkTpM = AlternatingTensorSpace(p, self.type, self.manifold)
    return ExteriorDerivativeTensor(self, TkTpM)

def exterior_derivative(w: DifferentialForm, brute_force: bool=False) -> DifferentialForm:
  """Compute the exterior derivative of a differential form.

  Args:
    w: The differential form

  Returns:
    dw
  """
  if (isinstance(w, CovectorField) == False) and (isinstance(w, DifferentialForm) == False):
    # Exterior derivative is same as exterior derivative for 0 tensors (functions)
    assert isinstance(w, Map)
    return FunctionDifferential(w)

  from src.connection import LieAlgebraValuedDifferentialForm
  if isinstance(w, LieAlgebraValuedDifferentialForm):
    # Apply the exterior derivative to each of the coefficients
    dws = [exterior_derivative(_w) for _w in w.ws]
    return LieAlgebraValuedDifferentialForm(dws, w.basis_vector_fields)

  if brute_force == False:
    # This is much faster than the brute force method
    return ExteriorDerivativeForm(w)
  else:
    # Brute force method
    coords_and_basis = w.get_coordinate_functions_with_basis()
    wI, eI = zip(*coords_and_basis)
    dwI = [FunctionDifferential(_wI) for _wI in wI]
    dw = None
    for i, (_dWI, _eI) in enumerate(zip(dwI, eI)):
      term = wedge_product_form(_dWI, _eI)
      dw = term if i == 0 else dw + term

    assert isinstance(dw, DifferentialForm)
    return dw

################################################################################################################

class PullbackDifferentialForm(DifferentialForm):
  """The pullback of a DifferentialForm field w through a map F

  Attributes:
    F: Map
    T: Tensor field
  """
  def __init__(self, F: Map, T: DifferentialForm):
    """Create a new pullback covariant tensor field

    Args:
      F: Map
      T: Tensor field
    """
    # T must be a covariant tensor field
    assert T.manifold == F.image
    assert T.type.k == 0
    self.T = T
    self.F = F
    super().__init__(tensor_type=T.type, M=F.domain)

  def apply_to_point(self, p: Point) -> Tensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    Tp = self.T(self.F(p))
    dFp_ = self.F.get_tensor_pullback(p, self.T.type)
    return dFp_(Tp)

def pullback_differential_form(F: Map, T: DifferentialForm) -> PullbackDifferentialForm:
  """The pullback of T by F.  F^*(T)

  Args:
    F: A map
    T: A covariant tensor field on defined on the image of F

  Returns:
    F^*(T)
  """
  return PullbackDifferentialForm(F, T)
