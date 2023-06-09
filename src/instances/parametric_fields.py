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
from src.vector import *
from src.manifold import *
from src.tangent import *
from src.cotangent import *
from src.tensor import *
from src.vector_field import *
from src.section import *
from src.riemannian_metric import *
from src.differential_form import *
from src.lie_algebra import *
from src.connection import *
import src.util as util

__all__ = ["AutonomousVectorField",
           "AutonomousCovectorField",
           "AutonomousTensorField",
           "ParametricRiemannianMetric",
           "ParametricDifferentialForm",
           "ParametricLieAlgebraValuedDifferentialForm",
           "AutonomousFrame",
           "AutonomousCoframe",
           "SimpleTimeDependentVectorField"]

class AutonomousVectorField(VectorField):
  """A vector field where we have a function to give us
  the coordinates of tangent vectors using whatever basis
  we get from charts of the manifold.  This is autonomous
  so doesn't depend on t.

  Attribues:
    vf: The function that gives us points
    M: Manifold
  """
  def __init__(self, vf: Callable[[Coordinate], Coordinate], M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      vf: Vector field coordinate function.
      M: The base manifold.
    """
    self.vf = vf
    self.manifold = M

    super().__init__(self.manifold)

    # Test that the vf is compatible with the manifold
    x = jnp.zeros(self.manifold.dimension)
    out = vf(x)
    assert out.shape == x.shape

  def apply_to_point(self, p: Point) -> TangentVector:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the vector field coordinates at p
    v_coords = self.vf(p_coords)

    # Construct the tangent vector
    TpM = TangentSpace(p, self.manifold)
    return TangentVector(v_coords, TpM)

class AutonomousCovectorField(CovectorField):
  """A covector field where we have a function to give us
  the coordinates of tangent vectors using whatever basis
  we get from charts of the manifold.  This is autonomous
  so doesn't depend on t.

  Attribues:
    vf: The function that gives us points
    M: Manifold
  """
  def __init__(self, vf: Callable[[Coordinate], Coordinate], M: Manifold):
    """Creates a new covector field.  Covector fields are sections of
    the tangent bundle.

    Args:
      vf: Covector field coordinate function.
      M: The base manifold.
    """
    self.vf = vf
    self.manifold = M

    super().__init__(self.manifold)

    # Test that the vf is compatible with the manifold
    x = jnp.zeros(self.manifold.dimension)
    out = vf(x)
    assert out.shape == x.shape

  def apply_to_point(self, p: Point) -> CotangentVector:
    """Evaluate the covector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent covector at p.
    """
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the covector field coordinates at p
    v_coords = self.vf(p_coords)

    # Construct the tangent covector
    coTpM = CotangentSpace(p, self.manifold)
    return CotangentVector(v_coords, coTpM)

################################################################################################################

class AutonomousFrame(Frame):
  """A frame implemented as the Jacobian of a function

  Attribues:
    vf: The function that gives us points
    M: Manifold
  """
  def __init__(self, vf: Callable[[Coordinate], Coordinate], M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      vf: Vector field coordinate function.
      M: The base manifold.
    """
    self.vf = jax.jacobian(vf)
    self.manifold = M
    super().__init__(self.manifold)

    # Test that the vf is compatible with the manifold
    x = jnp.zeros(self.manifold.dimension)
    out = self.vf(x)
    assert out.shape == (self.manifold.dimension, self.manifold.dimension)

  def apply_to_point(self, p: Point) -> TangentBasis:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the vector field coordinates at p
    v_coords = self.vf(p_coords)

    # To go back to the tangent basis, use the inverse of a local trivialization
    lt_map = self.frame_bundle.get_local_trivialization_map(p)
    return lt_map.inverse((p, v_coords))

class AutonomousCoframe(Coframe):
  """A frame implemented as the Jacobian of a function

  Attribues:
    vf: The function that gives us points
    M: Manifold
  """
  def __init__(self, vf: Callable[[Coordinate], Coordinate], M: Manifold):
    """Creates a new vector field.  Vector fields are sections of
    the tangent bundle.

    Args:
      vf: Vector field coordinate function.
      M: The base manifold.
    """
    self.vf = jax.jacobian(vf)
    self.manifold = M
    super().__init__(self.manifold)

    # Test that the vf is compatible with the manifold
    x = jnp.zeros(self.manifold.dimension)
    out = self.vf(x)
    assert out.shape == (self.manifold.dimension, self.manifold.dimension)

  def apply_to_point(self, p: Point) -> CotangentBasis:
    """Evaluate the vector field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tangent vector at p.
    """
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the vector field coordinates at p
    v_coords = self.vf(p_coords)

    # To go back to the tangent basis, use the inverse of a local trivialization
    lt_map = self.coframe_bundle.get_local_trivialization_map(p)
    return lt_map.inverse((p, v_coords))

################################################################################################################

class AutonomousTensorField(TensorField):
  """A tensor field where we have a function to give us
  the coordinates of tensors using whatever basis
  we get from charts of the manifold.  This is autonomous
  so doesn't depend on t.

  Attribues:
    tf: The function that gives us tensors at points
    M: Manifold
  """
  def __init__(self, tf: Callable[[Coordinate], Coordinate], tensor_type: TensorType, M: Manifold):
    """Creates a new tensor field.

    Args:
      tf: Tensor field coordinate function.
      M: The base manifold.
    """
    self.tf = tf
    self.tensor_type = tensor_type
    self.manifold = M
    super().__init__(tensor_type, self.manifold)

  def apply_to_point(self, p: Point) -> Tensor:
    """Evaluate the tensor field at a point.

    Args:
      p: Point on the manifold.

    Returns:
      Tensor at p.
    """
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the covector field coordinates at p
    v_coords = self.tf(p_coords)

    # Construct the tangent covector
    TkTpM = TensorSpace(p, self.tensor_type, self.manifold)
    return Tensor(v_coords, TkTpM=TkTpM)

################################################################################################################

class ParametricRiemannianMetric(TensorFieldToSymmetricTensorField):
  """A parametric Riemannian metric

  Attribues:
    tf: The function that gives us tensors at points
    M: Manifold
  """
  def __init__(self, tf: Callable[[Coordinate], Coordinate], M: Manifold):
    """Creates a new tensor field.

    Args:
      tf: Tensor field coordinate function.
      M: The base manifold.
    """
    self.manifold = M
    self.type = TensorType(0, 2)
    psd_tf = lambda x: tf(x).T@tf(x) # Need PSD coordinates
    T = AutonomousTensorField(psd_tf, self.type, self.manifold)
    super().__init__(T, self.type, self.manifold)

class ParametricDifferentialForm(TensorFieldToDifferentialForm):
  """A parametric differential form

  Attribues:
    tf: The function that gives us tensors at points
    M: Manifold
  """
  def __init__(self, tf: Callable[[Coordinate], Coordinate], tensor_type: TensorType, M: Manifold):
    """Creates a new tensor field.

    Args:
      tf: Tensor field coordinate function.
      M: The base manifold.
    """
    self.manifold = M
    T = AutonomousTensorField(tf, tensor_type, self.manifold)
    super().__init__(T, tensor_type, self.manifold)

class ParametricLieAlgebraValuedDifferentialForm(LieAlgebraValuedDifferentialForm):
  """A parametric lie algebra valued differential form

  Attribues:
    tf: The function that gives us tensors at points
    M: Manifold
  """
  def __init__(self, tfs: Callable[[Coordinate], Coordinate], tensor_type: TensorType, M: Manifold, lieG: LieAlgebra):
    """Creates a new tensor field.

    Args:
      tf: Tensor field coordinate function.
      M: The base manifold.
    """
    self.manifold = M
    assert isinstance(lieG, LieAlgebra)
    self.lieG = lieG
    self.basis_vector_fields = self.lieG.get_basis()

    assert len(tfs) == len(self.basis_vector_fields)

    self.tfs = tfs

    ws = [ParametricDifferentialForm(tf, tensor_type, M) for tf in self.tfs]

    super().__init__(ws, self.basis_vector_fields)

  # def apply_to_point(self, p: Point) -> LieAlgebraValuedAlternatingTensor:
  #   """Evaluate the tensor field at a point.

  #   Args:
  #     p: Point on the manifold.

  #   Returns:
  #     Tensor at p.
  #   """
  #   # Get the coordinates for the alternating tensor at each basis vector
  #   chart = self.manifold.get_chart_for_point(p)
  #   p_hat = chart(p)
  #   coords = self.tf(p_hat)

  #   # Make an alternating tensor out of each of the coordinates
  #   TkTpM = AlternatingTensorSpace(p, self.type, self.manifold)
  #   ws = [make_alternating(Tensor(x, TkTpM=TkTpM)) for x in coords]

  #   # Create the lie valued alternating tensor
  #   return LieAlgebraValuedAlternatingTensor(ws, basis=self.basis_vector_fields)

################################################################################################################

class SimpleTimeDependentVectorField(TimeDependentVectorField):
  # Construct a vector field using the charts for a manifold
  def __init__(self, vf, M: Manifold):
    self.vf = vf
    self.manifold = M

    # Test that the vf is compatible with the manifold
    x = jnp.zeros(self.manifold.dimension)
    out = vf(0.0, x)
    assert out.shape == x.shape

  def apply_to_point(self, t: Coordinate, p: Point) -> TangentVector:
    # Get the coordinates for p
    chart = self.manifold.get_chart_for_point(p)
    p_coords = chart(p)

    # Get the vector field coordinates at p
    v_coords = self.vf(t, p_coords)

    # Construct the tangent vector
    TpM = TangentSpace(p, self.manifold)
    return TangentVector(v_coords, TpM)