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
from src.instances.manifolds import EuclideanManifold
import src.util as util
import einops
import itertools
import abc
import math

__all__ = ["Tensor",
           "TensorType",
           "CovariantTensor",
           "TensorSpace",
           "TensorProductTensor",
           "TensorProductSpace",
           "tensor_space_product",
           "as_tensor",
           "tensor_product",
           "as_tensor_field",
           "TensorBundle",
           "TensorField",
           "TensorFieldProduct",
           "tensor_field_product",
           "PullbackOfTensor",
           "PullbackTensorField",
           "pullback_tensor_field"]

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
    self.xs = x
    self.TkTpM = TkTpM
    self.manifold = self.TkTpM.manifold
    self.phi = self.TkTpM.phi
    self.phi_inv = self.phi.get_inverse()
    self.p = self.TkTpM.p
    self.type = self.TkTpM.type

    self.contract = self.TkTpM.get_coordinate_indices()

  def to_tensor(self):
    x = self.get_dense_coordinates()
    return Tensor(x, TensorSpace(self.p, self.type, self.manifold))

  def get_dense_coordinates(self) -> Coordinate:
    """Get the dense representation of the coordinates

    Returns:
      A single array with the coordinates
    """
    contract = self.TkTpM.get_coordinate_indices()
    tokens, _ = util.extract_tokens(contract)
    contract = f"{contract} -> " + " ".join(tokens)
    return einops.einsum(*self.xs, contract)

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

  def get_coordinates(self, component_function: Chart) -> Coordinate:
    """Get the coordinates of this vector in terms of coordinates function.
    If J = dz/dx is the Jacobian of the component function and G = J^{-1},
    then the contravariant components of this tensor transform with J and
    the covariant transform with G.

    Args:
      component_function: A chart that gives coordinates

    Returns:
      Coordinates of this vector for component_function
    """
    if component_function == self.phi:
      return self.get_dense_coordinates()

    # We need to get the coorindates of the transition map
    F_hat = compose(component_function, self.phi.get_inverse())
    p_hat = self.phi(self.p)
    q_hat = component_function(self.p)

    # TODO: IS THERE A VJP/JVP THAT CAN SPEED THIS UP?
    dzdx = jax.jacobian(F_hat)(p_hat)
    dxdz = jax.jacobian(F_hat.get_inverse())(q_hat)

    # We need to apply the Jacobian of F to each coordinate axis.
    # Transpose dxdz so that the contraction makes sense.
    coordinate_transforms = [dzdx]*self.type.k + [dxdz.T]*self.type.l

    # Get the contraction corresponding to the tensor's coordinates
    # If this tensor is the product of a 1 and 2 covariant tensor,
    # this would be 'l0, l1 l2'
    coord_contract = self.TkTpM.get_coordinate_indices()

    # Get the unique tokens in the coordinate contract
    # In example, would be 'new_l0 new_l1 new_l2'
    tokens, _ = util.extract_tokens(coord_contract)
    output_tokens = ["new_"+t for t in tokens]
    output_contract = " ".join(output_tokens)

    # Each of the Jacobians transforms a single axis
    # In running example, would be 'new_l0 l0, new_l1 l1, new_l2 l2'
    jacobian_contract = ", ".join([f"new_{t} {t}" for t in tokens])

    # Put it all together
    # In example, would be 'l0, l1 l2, new_l0 l0, new_l1 l1, new_l2 l2 -> new_l0 new_l1 new_l2'
    contract = coord_contract + ", " + jacobian_contract + " -> " + output_contract

    # Get the coordinates of the output tensor.
    coords = einops.einsum(*self.xs, *coordinate_transforms, contract)
    return coords

  def __rmul__(self, a: float) -> "Tensor":
    """Multiply the tensor by a scalar

    Args:
      a: A scalar

    Returns:
      (aX)_p
    """
    assert a in Reals(dimension=1)
    # Only multiply one of the coordinates by a, otherwise
    # we'd be multiplying the dense coordinates by a more
    # than once!
    xs = (self.xs[0]*a,) + self.xs[1:]
    return Tensor(*xs, TkTpM=self.TkTpM)

  def __add__(self, Y: "Tensor") -> "Tensor":
    """Add two tensors together.

    Args:
      Y: Another tensor

    Returns:
      Xp + Yp
    """
    # Must be the same type
    assert isinstance(Y, Tensor)
    assert self.type == Y.type
    x_coords = self.get_dense_coordinates()
    y_coords = Y.get_coordinates(self.phi)

    # Need to create a new tensor space to get the correct
    # contraction indices
    TkTpM = TensorSpace(self.p, self.type, self.manifold)
    return Tensor(x_coords + y_coords, TkTpM=TkTpM)

  def __radd__(self, Y: "Tensor") -> "Tensor":
    """Add Y from the right

    Args:
      Y: Another tensor

    Returns:
      X + Y
    """
    assert isinstance(Y, Tensor)
    return self + Y

################################################################################################################

class TensorType(namedtuple("TensorType", ["k", "l"])):
  def __add__(self, other: "TensorType"):
    return TensorType(self.k + other.k, self.l + other.l)

class CovariantTensor(Tensor):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert self.type.k == 0

################################################################################################################

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

    self.dimension = self.manifold.dimension**(self.type.k + self.type.l)

    # Keep track of the chart function for the manifold
    self.phi = self.manifold.get_chart_for_point(self.p)

    super().__init__(dimension=self.dimension)

  @property
  def k_names(self) -> List[str]:
    """Get the indices for the coordinates that we'll plug into
    einsum when evaluating the tensor.

    Returns:
      If this has 3 contravariant indices, then returns "k0 k1 k2"
    """
    return [f"k{k}" for k in range(self.type.k)]

  @property
  def l_names(self) -> List[str]:
    """Get the indices for the coordinates that we'll plug into
    einsum when evaluating the tensor.

    Returns:
      If this has 3 covariant indices, then returns "l0 l1 l2"
    """
    return [f"l{l}" for l in range(self.type.l)]

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
    """Construct a tensor from an input of coordinates.

    Args:
      xs: Coordinates

    Returns:
      Tensor object with the coordinates "xs"
    """
    return Tensor(*xs, TkTpM=self)

  def get_atlas(self):
    """Return the atlas

    Returns:
      Atlas object
    """
    def chart_fun(v, inverse=False):
      assert 0, "Need to implement correctly and make modifications for symmetric and alternating spaces"
      contract = self.get_coordinate_indices()
      if inverse == False:
        assert isinstance(v, self.Element)
        return v.xs # Returns a list of coordinates
      else:
        xs = v
        return self.get_tensor_at(xs)

    self.chart = Chart(chart_fun, domain=self, image=Reals(dimension=self.dimension))
    return Atlas([self.chart])

  def get_basis(self, dual: Optional[bool]=False) -> List[Tensor]:
    """Get a basis of vectors for this tensor space

    Args:
      dual: Get the dual basis

    Returns:
      A list of tensors that form a basis for the tensor space
    """

    # Contravariant part first
    contravariant_bases = []
    for i in range(self.type.k):
      TpM = TangentSpace(self.p, self.manifold)
      basis = TpM.get_basis() if dual == False else TpM.get_dual_basis()
      contravariant_bases.append(basis)

    # Next, covariant part
    covariant_bases = []
    for i in range(self.type.l):
      coTpM = CotangentSpace(self.p, self.manifold)
      basis = coTpM.get_basis() if dual == False else coTpM.get_dual_basis()
      covariant_bases.append(basis)

    # A basis tensor consists of some combination of the
    # different basis (co)vectors for each space.
    basis = []
    for contravariant_basis in itertools.product(*contravariant_bases):
      for covariant_basis in itertools.product(*covariant_bases):
        basis_tensor = tensor_product(*contravariant_basis, *covariant_basis)
        basis.append(basis_tensor)

    assert len(basis) == self.dimension
    return basis

  def get_dual_basis(self) -> List[Tensor]:
    """Get a basis of vectors for this tensor space

    Returns:
      A list of tensors that form a basis for the tensor space
    """
    return self.get_basis(dual=True)

################################################################################################################

class TensorProductTensor(Tensor):
  """A tensor that comes from the tensor product of tensors

  Attributes:
    Ts: A list of tensors
  """
  def __init__(self, *Ts: List[Tensor]):
    """Creates a new tensor product tensor.

    Args:
      Ts: A list of tensors
    """
    self.Ts = Ts

    # Append the lists
    self.xs = reduce(lambda x, y: x + y, [T.xs for T in self.Ts])
    self.TkTpM = tensor_space_product(*[T.TkTpM for T in self.Ts])
    super().__init__(*self.xs, TkTpM=self.TkTpM)

  def decompose(self) -> List[Union[TangentVector,CotangentVector,Tensor]]:
    """Split a product of tensors back into their components and cast the
    types to tangent/cotangent vectors if possible.

    Returns:
      A list of tensors
    """
    new_Ts = []
    for T in self.Ts:
      if T.type == TensorType(1, 0):
        new_T = TangentVector(T.xs[0], TangentSpace(self.p, self.manifold))
      elif T.type == TensorType(0, 1):
        new_T = CotangentVector(T.xs[0], CotangentSpace(self.p, self.manifold))
      else:
        new_T = T
      new_Ts.append(new_T)
    return new_Ts

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

    # Extract the covectors and tangent vectors
    covectors = Xs[:self.type.k]
    for w in covectors:
      assert isinstance(w, CotangentVector)

    vectors = Xs[self.type.k:]
    for X in vectors:
      assert isinstance(X, TangentVector)

    # Apply each of the inputs to the tensors
    start_k = 0
    start_l = 0
    out = 1.0
    for T in self.Ts:
      end_k = start_k + T.type.k
      end_l = start_l + T.type.l
      out *= T(*covectors[start_k:end_k], *vectors[start_l:end_l])

      start_k, start_l = end_k, end_l

    return out

################################################################################################################

class TensorProductSpace(TensorSpace):
  """Tensor space of a manifold.  Is defined at a point, TkTpM

  Attributes:
    p: The point where the space lives.
    contract: Tells us what types of objects this tensor accepts
    M: The manifold.
  """
  Element = TensorProductTensor

  def __init__(self, *spaces: List[TensorSpace]):
    """Creates a new tensor product space

    Args:
      spaces: A list of tensor spaces
    """
    manifold = spaces[0].manifold
    assert all(S.manifold == manifold for S in spaces)

    self.manifold = manifold
    self.p = spaces[0].p

    self.spaces = spaces
    self.type = reduce(lambda x, y: x + y, [S.type for S in self.spaces])

    super().__init__(self.p, self.type, self.manifold)

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
    def increase_contraction(tensor_type: TensorType, contract: str):
      new_contract = ""
      contract_elements = contract.split(", ")
      for contract_element_indices in contract_elements:
        indices = contract_element_indices.split(" ")

        new_indices = ""
        for index in indices:
          if index[0] == "k":
            new_index = "k" + str(tensor_type.k + int(index[1:]))
          else:
            assert index[0] == "l"
            new_index = "l" + str(tensor_type.l + int(index[1:]))
          new_indices += new_index + " "

        # Trim the last space
        new_indices = new_indices[:-1]
        new_contract += new_indices + ","
      new_contract = new_contract[:-1]
      return new_contract

    contract = self.spaces[0].get_coordinate_indices()
    tensor_type = self.spaces[0].type
    for space in self.spaces[1:]:
      old_contract = space.get_coordinate_indices()
      contract += ", " + increase_contraction(tensor_type, old_contract)
      tensor_type += space.type

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
    assert 0, "implement this"
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

def tensor_space_product(*spaces: List[TensorSpace]) -> TensorSpace:
  """Tensor product of tensor spaces

  Args:
    spaces: A list of tensor spaces

  Returns:
    A new tensor of degree equal to the sum of that of the ones in spaces
  """
  return TensorProductSpace(*spaces)

################################################################################################################

def as_tensor(T: Union[CotangentVector,TangentVector]):
  """Turn a (co)vector into a tensor object

  Args:
    T: A (co)tangnet vector

  Returns:
    A tensor that is equivalent to T
  """
  if isinstance(T, CotangentVector):
    from src.differential_form import AlternatingTensorSpace, AlternatingTensor
    space = AlternatingTensorSpace(T.coTpM.p, TensorType(0, 1), T.coTpM.manifold)
    return AlternatingTensor(T.x, TkTpM=space)

  elif isinstance(T, TangentVector):
    space = TensorSpace(T.TpM.p, TensorType(1, 0), T.TpM.manifold)
    return Tensor(T.x, TkTpM=space)

  elif isinstance(T, Tensor):
    return T

  else:
    assert 0, "Invalid input"

def tensor_product(*Ts: List[Tensor]) -> Tensor:
  """Tensor product

  Args:
    Ts: A list of tensors

  Returns:
    A new tensor of degree equal to the sum of that of w and n
  """
  # Might need to case cotangent vectors as tensors
  Ts = [T if isinstance(T, Tensor) else as_tensor(T) for T in Ts]
  return TensorProductTensor(*Ts)

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
    self.dimension = self.manifold.dimension**(self.type.k + self.type.l)
    super().__init__(M, EuclideanManifold(dimension=self.dimension))

  def __contains__(self, x: Tensor) -> bool:
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
    type: The type of the tensor field
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

  def apply_to_co_vector_fields(self, *Xs: List[Union[VectorField,CovectorField]]) -> Map:
    """Evaluate the tensor field on (co)vector fields

    Args:
      Xs: A list of (co)vector fields

    Returns:
      A map over the manifold
    """
    def fun(p: Point):
      return self(p)(*[X(p) for X in Xs])
    return Map(fun, domain=self.manifold, image=Reals())

  @abc.abstractmethod
  def apply_to_point(self, p: Point) -> Tensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    pass

  def __call__(self, *x: Union[Point,List[Union[VectorField,CovectorField]]]) -> Union[Map,Tensor]:
    """Evaluate the tensor field at a point, or evaluate it on (co)vector fields

    Args:
      Xs: Either a point or a list of (co)vector fields

    Returns:
      Tensor at a point, or a map over the manifold
    """
    if isinstance(x[0], VectorField) or isinstance(x[0], CovectorField):
      return self.apply_to_co_vector_fields(*x)
    else:
      p = x[0]
      if util.GLOBAL_CHECK:
        assert p in self.manifold
      return self.apply_to_point(p)

  def __rmul__(self, f: Union[Map,float]) -> "TensorField":
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
      def __init__(self, T, pi, tensor_type):
        self.T = T
        self.type = tensor_type
        self.lhs = f
        self.is_float = f in Reals(dimension=1)
        Section.__init__(self, pi)

      def apply_to_co_vector_fields(self, *Xs: List[Union[VectorField,CovectorField]]) -> Map:
        return self.lhs*self.T(*Xs)

      def apply_to_point(self, p: Point) -> Tensor:
        fp = self.lhs if self.is_float else self.lhs(p)
        Tp = self.T(p)
        out = fp*Tp
        return out

    return SectionRHSProduct(self, self.pi, self.type)

  def __add__(self, Y: "TensorField") -> "TensorField":
    """Add two tensor fields together

    Args:
      Y: Another tensor field

    Returns:
      Xp + Yp
    """
    # Must be the same type
    assert isinstance(Y, TensorField) or isinstance(Y, FunctionDifferential)
    assert Y.type == self.type

    # class SumOfTensorFields(TensorField):
    class SumOfTensorFields(type(self)):
      def __init__(self, A: TensorField, B: TensorField, pi):
        self.A = A
        self.B = B
        self.type = A.type
        Section.__init__(self, pi)
        # super().__init__(self.type, A.manifold)

      def apply_to_co_vector_fields(self, *Xs: List[Union[VectorField,CovectorField]]) -> Map:
        return self.A(*Xs) + self.B(*Xs)

      def apply_to_point(self, p: Point) -> Tensor:
        Ap = self.A(p)
        Bp = self.B(p)
        return Ap + Bp

    return SumOfTensorFields(self, Y, self.pi)

  def __radd__(self, Y: "TensorField") -> "TensorField":
    """Add Y from the right

    Args:
      Y: Another tensor field

    Returns:
      X + Y
    """
    return self + Y

################################################################################################################

class TensorFieldProduct(TensorField):
  """A tensor field coming from a tensor product of tensor fields

  Attribues:
    Ts: A list of tensor fields
  """
  def __init__(self, *Ts: List[TensorField]):
    """Creates a new tensor field from a product of A and B

    Args:
      Ts: A list of tensor fields
    """
    self.manifold = Ts[0].manifold
    assert all(T.manifold == self.manifold for T in Ts)

    self.Ts = Ts
    self.type = reduce(lambda x, y: x + y, [T.type for T in self.Ts])

    super().__init__(self.type, self.manifold)

  def apply_to_co_vector_fields(self, *Xs: List[Union[VectorField,CovectorField]]) -> Map:
    """Evaluate the tensor field on vector/covector fields

    Args:
      Xs: A list of vector/covector fields

    Returns:
      A map over the manifold
    """
    # If we're evaluating at a list of vector fields, then pass the input fields to
    # each of the input tensor fields

    covector_fields = Xs[:self.type.k]
    vector_fields = Xs[self.type.k:]

    # Apply each of the input fields to the tensor fields
    start_k, start_l = 0, 0
    TXs = []
    for T in self.Ts:
      end_k = start_k + T.type.k
      end_l = start_l + T.type.l
      TX = T(*covector_fields[start_k:end_k], *vector_fields[start_l:end_l])
      TXs.append(TX)

      start_k, start_l = end_k, end_l

    def fun(p: Point):
      return reduce(lambda x, y: x(p)*y(p), TXs)

    return Map(fun, domain=self.manifold, image=Reals())

  def apply_to_point(self, p: Point) -> Tensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    # If we're evaluating at a point, then we evaluate the tensor product of the tensors
    return TensorProductTensor(*[T(p) for T in self.Ts])

def tensor_field_product(*Ts: List[TensorField]) -> TensorField:
  """Tensor product of tensor fields

  Args:
    Ts: A list of tensor fields

  Returns:
    Tensor product of the tensor fields
  """
  return TensorFieldProduct(*Ts)

################################################################################################################

def as_tensor_field(w: CovectorField):
  """Turn a covector field into a (0,k) tensor field

  Args:
    w: The covector field

  Returns:
    w but as a tensor field object
  """
  class CovectorFieldAsTensor(TensorField):
    def __init__(self, w):
      self.w = w
      tensor_type = TensorType(0, 1)
      super().__init__(tensor_type, self.w.manifold)

    def apply_to_point(self, p: Point) -> Tensor:
      wp = self.w.apply_to_point(p)
      return as_tensor(wp)

  return CovectorFieldAsTensor(w)

################################################################################################################

class PullbackOfTensor(LinearMap[CovariantTensor,CovariantTensor]):
  """The pullback map for a function that can be applied to a tensor

  Attributes:
    F: A map from M->N.
    p: The point where the differential is defined.
  """
  def __init__(self, F: Map[Input,Output], p: Point, tensor_type: TensorType):
    """Creates the differential of F at p

    Args:
    F: A map from M->N.
    p: The point where the differential is defined.
    """
    self.p = p
    self.q = F(self.p)
    self.F = F

    self.type = tensor_type

    self.TkTpM = TensorSpace(self.p, tensor_type=self.type, M=F.domain)
    self.TkTpN = TensorSpace(self.q, tensor_type=self.type, M=F.image)
    Map.__init__(self, f=self.__call__, domain=self.TkTpN, image=self.TkTpM)

  def __call__(self, T: CovariantTensor) -> CovariantTensor:
    """Apply the differential to a tangent vector.

    Args:
      T: A covariant tensor on N = F(M)

    Returns:
      dF_p^*(T)
    """
    assert T.type.k == 0

    # This is definitely not the most efficient way
    # TODO: IMPROVE EFFICIENCY USING VJP/JVP FOR TENSORS
    jacobian_matrix = self.F.get_pullback(self.p).get_coordinates()

    # We need to apply the Jacobian of F to each coordinate axis
    coordinate_transforms = [jacobian_matrix]*T.type.l

    # Get the contraction corresponding to the tensor's coordinates
    # If T is the product of a 1 and 2 covariant tensor, this would be
    # 'l0, l1 l2'
    coord_contract = T.TkTpM.get_coordinate_indices()

    # Get the unique tokens in the coordinate contract
    # In example, would be 'new_l0 new_l1 new_l2'
    tokens, _ = util.extract_tokens(coord_contract)
    output_tokens = ["new_"+t for t in tokens]
    output_contract = " ".join(output_tokens)

    # Each of the Jacobians transforms a single axis
    # In running example, would be 'new_l0 l0, new_l1 l1, new_l2 l2'
    jacobian_contract = ", ".join([f"new_{t} {t}" for t in tokens])

    # Put it all together
    # In example, would be 'l0, l1 l2, new_l0 l0, new_l1 l1, new_l2 l2 -> new_l0 new_l1 new_l2'
    contract = coord_contract + ", " + jacobian_contract + " -> " + output_contract

    # Get the coordinates of the output tensor.  This won't factor like
    # the coordinates of T might.
    output_coords = einops.einsum(*T.xs, *coordinate_transforms, contract)

    # Create a new output tensor that has the same type as T (in case T is alternating)
    from src.differential_form import AlternatingTensor, AlternatingTensorSpace
    if isinstance(T, AlternatingTensor):
      TkTpM = AlternatingTensorSpace(self.p, tensor_type=self.type, M=self.F.domain)
      return AlternatingTensor(output_coords, TkTpM=TkTpM)
    else:
      TkTpM = TensorSpace(self.p, tensor_type=self.type, M=self.F.domain)
      return Tensor(output_coords, TkTpM=TkTpM)

################################################################################################################

class PullbackTensorField(TensorField):
  """The pullback of a tensorfield field w through a map F

  Attributes:
    F: Map
    T: Tensor field
  """
  def __init__(self, F: Map, T: TensorField):
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

def pullback_tensor_field(F: Map, T: TensorField) -> PullbackTensorField:
  """The pullback of T by F.  F^*(T)

  Args:
    F: A map
    T: A covariant tensor field on defined on the image of F

  Returns:
    F^*(T)
  """
  return PullbackTensorField(F, T)

################################################################################################################
