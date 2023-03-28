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
from src.instances.manifolds import EuclideanManifold
import src.util as util
import einops
import itertools
import abc
import math

__all__ = ["Tensor",
           "TensorType",
           "TensorSpace",
           "TensorProductTensor",
           "TensorProductSpace",
           "tensor_space_product",
           "SymmetrizedTensor",
           "symmetrize",
           "symmetric_product",
           "TensorBundle",
           "TensorField",
           "TensorFieldProduct",
           "tensor_field_product"]

class Tensor(MultlinearMap[List[Union[CotangentVector,TangentVector]],Coordinate]):
  """A tensor is a real valued multilinear function on one or more variables.

  Attributes:
    x: The coordinates of the vector in the basis induced by a chart.
    TkTpM: The space of mixed type tensors that this tensor lives on
  """
  def __init__(self, *x: Coordinate, TkTpM: "TensorSpace"):
    """Creates a new tensor.

    Args:
      xs: A list of independent coordinates to make tensor
      TkTpM: The tangent space that the tensor lives on.
    """
    # assert x.ndim == TkTpM.coordinate_dim
    self.xs = x
    self.TkTpM = TkTpM
    self.phi = self.TkTpM.phi
    self.phi_inv = self.phi.get_inverse()
    self.p = self.TkTpM.p
    self.type = self.TkTpM.type

    self.contract = self.TkTpM.get_coordinate_indices()

  def __call__(self, *Xs: List[Union[CotangentVector,TangentVector]]) -> Coordinate:
    """Apply the tensor to the inputs.

    Args:
      Xs: A list of cotangent vectors followed by tangent vectors

    Returns:
      Application of tensor to input covectors/vectors
    """
    assert len(Xs) == self.type.k + self.type.l

    # Check that the types of the inputs are correct and also
    # get the coordinates of the inputs
    args = []
    for w in Xs[:self.type.k]:
      assert isinstance(w, CotangentVector)
      coords = w.get_coordinates(self.phi)
      args.append(coords)

    for X in Xs[self.type.k:]:
      assert isinstance(X, TangentVector)
      coords = X.get_coordinates(self.phi)
      args.append(coords)

    # Get the new coordinates
    contract = self.contract
    if len(self.TkTpM.k_names) > 0:
      contract += ", "
      contract += ", ".join(self.TkTpM.k_names)
      if len(self.TkTpM.l_names) > 0:
        contract += ", "
    if len(self.TkTpM.l_names) > 0:
      if len(self.TkTpM.k_names) == 0:
        contract += ", "
      contract += ", ".join(self.TkTpM.l_names)
    contract += " ->"
    return einops.einsum(*self.xs, *args, contract)

  def __rmul__(self, a: float) -> "Tensor":
    """Proposition 12.22

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    assert a in Reals(dimension=1)
    xs = [x*a for x in self.xs]
    return Tensor(*xs, TkTpM=self.TkTpM)

################################################################################################################

class TensorType(namedtuple("TensorType", ["k", "l"])):
  def __add__(self, other: "TensorType"):
    return TensorType(self.k + other.k, self.l + other.l)

class TensorSpace(VectorSpace):
  """Tensor space of a manifold.  Is defined at a point, TkTpM

  Attributes:
    p: The point where the space lives.
    contract: Tells us what types of objects this tensor accepts
    M: The manifold.
  """
  Element = Tensor

  def __init__(self, p: Point, tensor_type: TensorType, M: Manifold):
    """Creates a new space of mixed tensors on the tangent bundle

    Args:
      p: The point where the space lives.
      type: The tensor type (k,l)
      M: The manifold.
    """
    assert isinstance(tensor_type, TensorType)
    self.p = p
    self.type = tensor_type
    self.manifold = M

    # Need to keep track of the shape of our coordinates
    self.coordinate_dim = self.type.k + self.type.l
    self.dimension = self.manifold.dimension**(self.coordinate_dim)

    # Keep track of the chart function for the manifold
    self.phi = self.manifold.get_chart_for_point(self.p)

    super().__init__(dimension=self.dimension)

    # These are the letters that we'll use in our einsum
    # corresponding to the number of dimensions each covariant
    # and contravariant index
    self.k_names = [f"k{k}" for k in range(self.type.k)]
    self.l_names = [f"l{l}" for l in range(self.type.l)]

  def get_coordinate_indices(self) -> str:
    """Get the indices for the coordinates that we'll plug into
    einsum when evaluating the tensor.

    Returns:
      A (3, 5) tensor will have "k0 k1 k2 l0 l1 l2 l3 l4"
    """
    if len(self.k_names) == 0:
      return " ".join(self.l_names)
    if len(self.l_names) == 0:
      return " ".join(self.k_names)
    return " ".join(self.k_names) + " " + " ".join(self.l_names)

  def get_tensor_at(self, xs):
    return Tensor(*xs, TkTpM=self)

  def get_atlas(self):
    """Return the atlas

    Returns:
      Atlas object
    """
    def chart_fun(v, inverse=False):
      contract = self.get_coordinate_indices()
      if inverse == False:
        assert isinstance(v, self.Element)
        return v.xs # Returns a list of coordinates
      else:
        xs = v
        return self.get_tensor_at(xs)

    self.chart = Chart(chart_fun, domain=self, image=Reals(dimension=self.dimension))
    return Atlas([self.chart])

################################################################################################################

class TensorProductTensor(Tensor):
  """A tensor that comes from the tensor product of two tensors

  Attributes:
    a: Tensor a
    b: Tensor b
  """
  def __init__(self, a: Tensor, b: Tensor):
    """Creates a new tensor product tensor.

    Args:
      a: Tensor a
      b: Tensor b
    """
    self.a = a
    self.b = b

    # Append the lists
    self.xs = a.xs + b.xs
    self.TkTpM = tensor_space_product(self.a.TkTpM, self.b.TkTpM)
    super().__init__(*self.xs, TkTpM=self.TkTpM)

  def call_unvectorized(self, *Xs: List[Union[CotangentVector,TangentVector]]) -> Coordinate:
    """Apply the tensor to the inputs by applying the vectors to each tensor.
    The implementation from the Tensor class should work the same.

    Args:
      Xs: A list of tangent vectors followed by covectors

    Returns:
      Application of tensor to input covectors/vectors
    """
    # Apply the cotangent vectors first and then the tangent vectors
    assert len(Xs) == self.type.k + self.type.l

    # The first a.k elements of Xs should be the covectors for a
    start = 0
    end = self.a.type.k
    for i, w in enumerate(Xs[start:end]):
      assert isinstance(w, CotangentVector)

    # The next b.k elements of Xs should be the covectors for b
    start = end
    end += self.b.type.k
    for i, w in enumerate(Xs[start:end]):
      assert isinstance(w, CotangentVector)

    # The next a.l elements of Xs should be the vectors for a
    start = end
    end += self.a.type.l
    for i, w in enumerate(Xs[start:end]):
      assert isinstance(w, TangentVector)

    # The next b.l elements of Xs should be the vectors for b
    start = end
    end += self.b.type.l
    for i, w in enumerate(Xs[start:end]):
      assert isinstance(w, TangentVector)

    covectors = Xs[:self.type.k]
    vectors = Xs[self.type.k:]

    out1 = self.a(*covectors[:self.a.type.k], *vectors[:self.a.type.l])
    out2 = self.b(*covectors[self.a.type.k:], *vectors[self.a.type.l:])

    return out1*out2

################################################################################################################

class TensorProductSpace(TensorSpace):
  """Tensor space of a manifold.  Is defined at a point, TkTpM

  Attributes:
    p: The point where the space lives.
    contract: Tells us what types of objects this tensor accepts
    M: The manifold.
  """
  Element = TensorProductTensor

  def __init__(self, A: TensorSpace, B: TensorSpace):
    """Creates a new tensor product space

    Args:
      A: First tensor
      B: Second tensor
    """
    assert A.manifold == B.manifold
    self.A = A
    self.B = B
    self.type = A.type + B.type

    super().__init__(self.A.p, self.type, self.A.manifold)

  def get_coordinate_indices(self) -> str:
    """A (2, 3) tensor and (2, 2) tensor have indices
       "k0 k1 l0 l1 l2" and "k0 k1 l0 l1", the product's
       will be "k0 k1 l0 l1 l2, k2 k3 l3 l4".  If a is the
       tensor product of (2, 3) = (1, 2) + (1, 1) and b's
       type is (2, 1), the product would be
       "k0 l0 l1, k1 l2, k2 k3 l3"

    Returns:
      Coordinate indices
    """
    a_contract = self.A.get_coordinate_indices()
    b_contract = self.B.get_coordinate_indices()

    new_b_contract = ""
    b_contract_elements = b_contract.split(", ")
    for b_contract_element_indices in b_contract_elements:
      indices = b_contract_element_indices.split(" ")

      new_indices = ""
      for index in indices:
        if index[0] == "k":
          new_index = "k" + str(self.A.type.k + int(index[1:]))
        else:
          assert index[0] == "l"
          new_index = "l" + str(self.A.type.l + int(index[1:]))
        new_indices += new_index + " "

      # Trim the last space
      new_indices = new_indices[:-1]
      new_b_contract += new_indices + ","
    new_b_contract = new_b_contract[:-1]

    contract = a_contract + ", " + new_b_contract
    return contract

  def get_tensor_at(self, xs):
    """Gets the tensor product object of the list of input coordinates.
    If the original tensor was a nested tensor product, then might not
    reconstruct nested structure.  But this doesn't matter because of
    associativity of the tensor product.

    Args:
      xs: A list of coordinates for each tensor in a tensor product

    Returns:
      TensorProductTensor object
    """
    contract = self.get_coordinates()

    x_contracts = contract.split(", ")
    assert len(x_contracts) == len(xs)

    tensor_product = None
    for a_idx, b_idx, xa, xb in zip(x_contracts, x_contracts[1:], xs, xs[1:]):
      if tensor_product is None:
        a = Tensor(xa, TkTpM=self)
        b = Tensor(xb, TkTpM=self)
        tensor_product = TensorProductTensor(a, b)
        import pdb; pdb.set_trace()
      else:
        b = Tensor(xb, TkTpM=self)
        tensor_product = TensorProductTensor(tensor_product, b)

    return tensor_product

def tensor_space_product(A: TensorSpace, B: TensorSpace) -> TensorSpace:
  """Tensor product of tensor spaces

  Args:
    w: Tensor 1
    n: Tensor 2

  Returns:
    A new tensor of degree equal to the sum of that of w and n
  """
  assert A.manifold == B.manifold
  if util.GLOBAL_CHECK:
    assert A.p == B.p

  return TensorProductSpace(A, B)

################################################################################################################

def as_tensor(w: CotangentVector):
  """Turn a cotangent vector into a tensor object

  Args:
    w: A cotangnet vector

  Returns:
    A tensor that is equivalent to w
  """
  assert isinstance(w, CotangentVector)
  space = TensorSpace(w.coTpM.p, TensorType(0, 1), w.coTpM.manifold)
  return Tensor(w.x, TkTpM=space)

def tensor_product(w: Tensor, n: Tensor) -> Tensor:
  """Tensor product

  Args:
    w: Tensor 1
    n: Tensor 2

  Returns:
    A new tensor of degree equal to the sum of that of w and n
  """
  # Try casting w and n as tensors in the event that they are cotangent or tangent vectors
  if isinstance(w, CotangentVector):
    w = as_tensor(w)
  if isinstance(n, CotangentVector):
    n = as_tensor(n)

  return TensorProductTensor(w, n)

################################################################################################################

class SymmetrizedTensor(Tensor):
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
    sym_TkTpM = TensorSpace(self.T.p, self.T.type, self.T.TkTpM.manifold)
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

def symmetrize(T: Tensor):
  """Symmetrize a tensor.

  Args:
    T: Tensor to symmetrize

  Returns:
    Symmetric version of T
  """
  return SymmetrizedTensor(T)

def symmetric_product(a: Tensor, b: Tensor):
  """Symmetric product of 2 tensors

  Args:
    a: Tensor a
    b: Tensor b

  Returns:
    Symmetric product of a and b
  """
  return symmetrize(tensor_product(a, b))

################################################################################################################

class TensorBundle(FiberBundle):
  """Bundle of mixed tensor bundle

  Attributes:
    M: The manifold.
  """
  Element = Tensor # The tensor is evaluated at p!

  def __init__(self, tensor_type: TensorType, M: Manifold):
    """Creates a new tensor bundle

    Args:
      type: The tensor type (k,l)
      M: The manifold.
    """
    # The fiber of a tensor bundle is a tensor
    self.manifold = M
    self.type = tensor_type
    self.coordinate_dim = self.type.k + self.type.l
    self.dimension = self.manifold.dimension**(self.coordinate_dim)
    super().__init__(M, EuclideanManifold(dimension=self.dimension))

  def get_projection_map(self) -> Map[Tensor,Point]:
    """Get the projection map that goes from the total space
    to the base space

    Returns:
      The map x -> p, x in E, p in M
    """
    return Map(lambda v: v.p, domain=self, image=self.manifold)

  def get_local_trivialization_map(self, T: Tensor) -> Map[Tensor,Tuple[Point,Coordinate]]:
    """Goes to the product space representation at p so that
    local_trivialization*proj_1 = self.pi.

    Args:
      p: A point on the base manifold

    Returns:
      A mapping from the bundle to a product bundle that is locally the same.
    """
    assert isinstance(T, Tensor)

    product_dimensions = [x.size for x in T.xs]
    Ms = [EuclideanManifold(dimension=dim) for dim in product_dimensions]
    fiber = manifold_carteisian_product(*Ms)

    image = ProductBundle(self.manifold, fiber)
    def Phi(v, inverse=False):
      if inverse == False:
        return v.p, v.xs
      else:
        p, xs = v
        return Tensor(*xs, TkTpM=TensorSpace(p, self.tensor_type, self.manifold))
    return Diffeomorphism(Phi, domain=self, image=image)

  def get_atlas(self):
    """Computations are done using local trivializations, so this
    shouldn't matter.
    """
    return None

################################################################################################################

from src.section import Section
class TensorField(Section[Point,Tensor], abc.ABC):
  """Section of tensor bundle

  Attributes:
    X: Function that assigns a tensor to every point on the manifold
    M: The manifold that the tensor field is defined on
  """
  def __init__(self, tensor_type: TensorType, M: Manifold):
    """Creates a new tensor field.

    Args:
      type: The tensor type (k,l)
      M: The base manifold.
    """
    self.type = tensor_type
    self.manifold = M

    # Construct the bundle
    domain = M
    image = TensorBundle(tensor_type, M)
    pi = ProjectionMap(idx=0, domain=image, image=domain)
    super().__init__(pi)

  def __call__(self, *Xs: Union[Point,List[Union[VectorField,CovectorField]]]) -> Union[Map,Tensor]:
    """Evaluate the tensor field at a point, or evaluate it on vector/covector fields

    Args:
      Xs: Either a point or a list of vector/covector fields

    Returns:
      Tensor at a point, or a map over the manifold
    """
    def fun(p: Point):
      return self(p)(*[X(p) for X in Xs])
    return Map(fun, domain=self.manifold, image=Reals())

################################################################################################################

class TensorFieldProduct(TensorField):
  """A tensor field coming from a tensor product of two tensor fields

  Attribues:
    A: Tensor field 1
    B: Tensor field 2
  """
  def __init__(self, A: TensorField, B: TensorField):
    """Creates a new tensor field from a product of A and B

    Args:
      A: Tensor field 1
      B: Tensor field 2
    """
    assert A.manifold == B.manifold
    self.A = A
    self.B = B
    self.type = A.type + B.type

    super().__init__(self.type, A.manifold)

  def __call__(self, *Xs: Union[Point,List[Union[VectorField,CovectorField]]]) -> Union[Map,Tensor]:
    """Evaluate the tensor field at a point, or evaluate it on vector/covector fields

    Args:
      Xs: Either a point or a list of vector/covector fields

    Returns:
      Tensor at a point, or a map over the manifold
    """
    if isinstance(Xs[0], VectorField) or isinstance(Xs[0], CovectorField):
      # If we're evaluating at a list of vector fields, then pass the input fields to
      # each of the input tensor fields
      def fun(p: Point):
        Ap, Bp = self.A(p), self.B(p)

        covector_fields = Xs[:self.type.k]
        vector_fields = Xs[self.type.k:]

        covectors = [w(p) for w in covector_fields]
        vectors = [X(p) for X in vector_fields]

        out1 = Ap(*covectors[:self.A.type.k], *vectors[:self.A.type.l])
        out2 = Bp(*covectors[self.A.type.k:], *vectors[self.A.type.l:])
        return out1*out2

      return Map(fun, domain=self.manifold, image=Reals())
    else:
      # If we're evaluating at a point, then we evaluate the tensor product of the tensors
      p = Xs[0]
      Ap = self.A(p)
      Bp = self.B(p)
      return TensorProductTensor(Ap, Bp)

def tensor_field_product(A: TensorField, B: TensorField) -> TensorField:
  """Tensor product of tensor fields

  Args:
    A: Tensor field 1
    B: Tensor field 2

  Returns:
    Tensor product of A and B
  """
  return TensorFieldProduct(A, B)

################################################################################################################
