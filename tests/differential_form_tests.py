from functools import partial
from typing import Callable
from typing import NewType
import itertools
import src.util
import jax
import jax.numpy as jnp
import jax.random as random
from src.set import *
from src.map import *
from src.tangent import *
from src.cotangent import *
from src.manifold import *
from src.vector_field import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
from src.instances.vector_fields import *
from src.section import *
from src.tensor import *
from src.lie_derivative import *
from src.differential_form import *
import nux
import src.util as util
from tests.vector_field_tests import get_vector_field_fun
from tests.tensor_tests import get_tensor_field_fun
from tests.cotangent_tests import get_chart_fun

def get_inputs(dimension=5):
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=dimension)
  p = random.normal(rng_key, (M.dimension + 1,)); p = p/jnp.linalg.norm(p)

  # Evaluate it on some cotangent and tangent vectors
  tensor_type = TensorType(0, 3)
  TkTpM = TensorSpace(p, tensor_type, M)
  coords = random.normal(rng_key, ([M.dim]*sum(tensor_type)))
  T = Tensor(coords, TkTpM=TkTpM)

  # Evaluate it on some cotangent and tangent vectors
  tensor_type = TensorType(0, 5)
  TkTpM = TensorSpace(p, tensor_type, M)
  coords = random.normal(rng_key, ([M.dim]*sum(tensor_type)))
  T2 = Tensor(coords, TkTpM=TkTpM)

  # Now try a tensor product
  rng_key, _ = random.split(rng_key, 2)
  tangent_coords = random.normal(rng_key, (5, M.dim))
  TpM = TangentSpace(p, M)
  v1 = TangentVector(tangent_coords[0], TpM)
  v2 = TangentVector(tangent_coords[1], TpM)
  v3 = TangentVector(tangent_coords[2], TpM)
  v4 = TangentVector(tangent_coords[3], TpM)
  v5 = TangentVector(tangent_coords[4], TpM)

  return rng_key, M, p, T, T2, v1, v2, v3, v4, v5

################################################################################################################

def alternating_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs()

  # Check symmetrization
  AltT = make_alternating(T)
  out1 = AltT(v1, v2, v3)
  out2 = -AltT(v1, v3, v2)
  out3 = -AltT(v2, v1, v3)
  out4 = AltT(v2, v3, v1)
  out5 = AltT(v3, v1, v2)
  out6 = -AltT(v3, v2, v1)
  assert jnp.allclose(out1, out2)
  assert jnp.allclose(out1, out3)
  assert jnp.allclose(out1, out4)
  assert jnp.allclose(out1, out5)
  assert jnp.allclose(out1, out6)

  out1_ = AltT.call_unvectorized(v1, v2, v3)
  out2_ = -AltT.call_unvectorized(v1, v3, v2)
  out3_ = -AltT.call_unvectorized(v2, v1, v3)
  out4_ = AltT.call_unvectorized(v2, v3, v1)
  out5_ = AltT.call_unvectorized(v3, v1, v2)
  out6_ = -AltT.call_unvectorized(v3, v2, v1)
  assert jnp.allclose(out1_, out1)
  assert jnp.allclose(out2_, out2)
  assert jnp.allclose(out3_, out3)
  assert jnp.allclose(out4_, out4)
  assert jnp.allclose(out5_, out5)
  assert jnp.allclose(out6_, out6)

  # Check that repeating inputs gives 0
  out = AltT(v1, v2, v2)
  assert jnp.allclose(out, 0.0)

  # Check that linearly dependent inputs gives 0
  v4 = v1 + v2
  out = AltT(v1, v2, v4)
  assert jnp.allclose(out, 0.0)

  # Check that T is alternating iff Alt(T) is alternating
  coords = AltT.get_dense_coordinates()
  tensor_type = TensorType(0, 3)
  TkTpM = AlternatingTensorSpace(p, tensor_type, M)
  alpha = Tensor(coords, TkTpM=TkTpM)
  Alt_alpha = make_alternating(alpha)

  out1 = alpha(v1, v2, v3)
  out2 = Alt_alpha(v1, v2, v3)
  assert jnp.allclose(out1, out2)
  out = alpha(v1, v2, v3)

  # Check that we can get the dense coordinates correctly
  coords = AltT.get_dense_coordinates()
  check = AltT.xs[0]
  assert jnp.allclose(coords, check)

################################################################################################################

def elementary_alternating_tensor_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs()

  k = 3
  tensor_type = TensorType(0, k)
  TkTpM = AlternatingTensorSpace(p, tensor_type, M)

  # Apply the elementary alternating tensor to some tangent vectors
  I = MultiIndex(list(range(k)))
  eI = ElementaryAlternatingTensor(I, TkTpM)

  out1 = eI(v1, v2, v3)
  out2 = eI.call_unvectorized(v1, v2, v3)
  assert jnp.allclose(out1, out2)

  out1 = eI(v3, v2, v1)
  out2 = eI.call_unvectorized(v3, v2, v1)
  assert jnp.allclose(out1, out2)

  # Make sure that the elementary tensors are alternating
  AlteI = make_alternating(eI)
  coords1 = eI.get_dense_coordinates()
  coords2 = AlteI.get_dense_coordinates()
  assert jnp.allclose(coords1, coords2)

  # Try applying it to basis vectors
  J = MultiIndex(list(range(k)))
  basis_vectors = TangentSpace(TkTpM.p, TkTpM.manifold).get_basis()
  basis_vectors = [basis_vectors[j] for j in J]

  out1 = eI(*basis_vectors)
  out2 = kronocker_delta(I, J)

  assert jnp.allclose(out1, out2)

  # Check that we can decompose alternating tensors into their
  # basis elements correctly
  AltT = make_alternating(T)
  out = AltT.get_basis_elements()

  T_reconstr = None
  for wI, eI in out:
    term = wI*eI
    if T_reconstr is None:
      T_reconstr = term
    else:
      T_reconstr += term

  assert jnp.allclose(AltT.get_dense_coordinates(), T_reconstr.get_dense_coordinates())

################################################################################################################

def determinant_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs()

  # Create an alternating tensor
  AltT = make_alternating(T2)

  # Get a linear transformation of basis vectors
  A = random.normal(rng_key, (M.dimension + 1, M.dimension + 1))
  Q, _ = jnp.linalg.qr(A)
  def _F(x, inverse=False):
    if inverse == False:
      return Q@x
    return Q.T@x

  F = Diffeomorphism(_F, domain=M, image=M)
  q = F(p)

  dFp = F.get_differential(p)

  # Proposition 14.9
  out1 = AltT(dFp(v1), dFp(v2), dFp(v3), dFp(v4), dFp(v5))
  out2 = dFp.determinant()*AltT(v1, v2, v3, v4, v5)

  assert jnp.allclose(out1, out2)

################################################################################################################

def wedge_product_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs()
  k1, k2, k3, k4 = random.split(rng_key, 4)

  # Lemma 14.10. Check that we can wedge basis tensors to get other basis tensors
  tensor_type1 = TensorType(0, 1)
  TkTpM1 = AlternatingTensorSpace(p, tensor_type1, M)
  coords1 = random.normal(rng_key, ([M.dim]*sum(tensor_type1)))
  w1 = make_alternating(Tensor(coords1, TkTpM=TkTpM1))

  tensor_type2 = TensorType(0, 2)
  TkTpM2 = AlternatingTensorSpace(p, tensor_type2, M)
  coords2 = random.normal(rng_key, ([M.dim]*sum(tensor_type2)))
  w2 = make_alternating(Tensor(coords2, TkTpM=TkTpM2))

  tensor_type3 = TensorType(0, 3)
  TkTpM3 = AlternatingTensorSpace(p, tensor_type3, M)

  I1 = MultiIndex([1])
  I2 = MultiIndex([0, 3])
  I3 = MultiIndex([1, 0, 3])

  eI1 = ElementaryAlternatingTensor(I1, TkTpM1)
  eI2 = ElementaryAlternatingTensor(I2, TkTpM2)
  eI3 = ElementaryAlternatingTensor(I3, TkTpM3)

  check = wedge_product(eI1, eI2)

  coords1 = eI3.get_dense_coordinates()
  coords2 = check.get_dense_coordinates()
  assert jnp.allclose(coords1, coords2)

  # Proposition 14.11.  Verify properties of the wedge product
  a1, a2 = random.normal(rng_key, (2,))

  tensor_type1 = TensorType(0, 2)
  TkTpM1 = AlternatingTensorSpace(p, tensor_type1, M)
  coords1 = random.normal(rng_key, ([M.dim]*sum(tensor_type1)))
  eta1 = make_alternating(Tensor(coords1, TkTpM=TkTpM1))

  tensor_type2 = TensorType(0, 2)
  TkTpM2 = AlternatingTensorSpace(p, tensor_type2, M)
  coords2 = random.normal(rng_key, ([M.dim]*sum(tensor_type2)))
  eta2 = make_alternating(Tensor(coords2, TkTpM=TkTpM2))

  # a)
  test1 = wedge_product(a1*eta2 + a2*w2, eta1)
  test2 = a1*wedge_product(eta2, eta1) + a2*wedge_product(w2, eta1)
  assert jnp.allclose(test1.get_dense_coordinates(), test2.get_dense_coordinates())

  test1 = wedge_product(eta1, a1*eta2 + a2*w2)
  test2 = a1*wedge_product(eta1, eta2) + a2*wedge_product(eta1, w2)
  assert jnp.allclose(test1.get_dense_coordinates(), test2.get_dense_coordinates())

  # b)
  test1 = wedge_product(wedge_product(w1, w2), eta1)
  test2 = wedge_product(eta1, wedge_product(w1, w2))
  test3 = wedge_product(w1, w2, eta1)
  coords1 = test1.get_dense_coordinates()
  coords2 = test2.get_dense_coordinates()
  coords3 = test3.get_dense_coordinates()
  assert jnp.allclose(coords1, coords2)
  assert jnp.allclose(coords1, coords3)

  # c)
  test1 = wedge_product(w1, w2)
  test2 = wedge_product(w2, w1)
  assert jnp.allclose(test1.get_dense_coordinates(), test2.get_dense_coordinates())

  # d) Use all of the dual basis vectors except for index 2
  e0, e1, e2, e3, e4 = TangentSpace(p, M).get_dual_basis()
  I = MultiIndex([0, 3, 1, 4])
  test1 = wedge_product(e0, e3, e1, e4)

  TkTpM = AlternatingTensorSpace(p, TensorType(0, 4), M)
  test2 = ElementaryAlternatingTensor(I, TkTpM)
  coords1 = test1.get_dense_coordinates()
  coords2 = test2.get_dense_coordinates()
  assert jnp.allclose(coords1, coords2)

  # e)
  cotangent_coords = random.normal(rng_key, (5, M.dim))
  coTpM = CotangentSpace(p, M)
  w1 = CotangentVector(cotangent_coords[0], coTpM)
  w2 = CotangentVector(cotangent_coords[1], coTpM)
  w3 = CotangentVector(cotangent_coords[2], coTpM)
  w4 = CotangentVector(cotangent_coords[3], coTpM)
  w5 = CotangentVector(cotangent_coords[4], coTpM)

  out1 = wedge_product(w1, w2, w3, w4, w5)(v1, v2, v3, v4, v5)
  vs = [v1, v2, v3, v4, v5]
  ws = [w1, w2, w3, w4, w5]
  mat = [[w(v) for w in ws] for v in vs]
  out2 = jnp.linalg.det(jnp.array(mat))

  assert jnp.allclose(out1, out2)

################################################################################################################

def interior_product_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs()

  # Try using the interior product
  AltT = make_alternating(T)
  v1_into_AltT = interior_product(AltT, v1)
  out1 = v1_into_AltT(v2, v3)
  out2 = AltT(v1, v2, v3)
  assert jnp.allclose(out1, out2)

  # Lemma 14.13
  tensor_type1 = TensorType(0, 2)
  TkTpM1 = AlternatingTensorSpace(p, tensor_type1, M)
  coords1 = random.normal(rng_key, ([M.dim]*sum(tensor_type1)))
  w1 = make_alternating(Tensor(coords1, TkTpM=TkTpM1))

  tensor_type2 = TensorType(0, 3)
  TkTpM2 = AlternatingTensorSpace(p, tensor_type2, M)
  coords2 = random.normal(rng_key, ([M.dim]*sum(tensor_type2)))
  w2 = make_alternating(Tensor(coords2, TkTpM=TkTpM2))

  v = v1
  test1 = interior_product(wedge_product(w1, w2), v)
  test2 = wedge_product(interior_product(w1, v), w2) + (-1.0)**w1.type.l*wedge_product(w1, interior_product(w2, v))
  coords1 = test1.get_dense_coordinates()
  coords2 = test2.get_dense_coordinates()
  assert jnp.allclose(coords1, coords2)

  # Check Eq. 14.12
  cotangent_coords = random.normal(rng_key, (3, M.dim))
  coTpM = CotangentSpace(p, M)
  w1 = CotangentVector(cotangent_coords[0], coTpM)
  w2 = CotangentVector(cotangent_coords[1], coTpM)
  w3 = CotangentVector(cotangent_coords[2], coTpM)

  test1 = interior_product(wedge_product(w1, w2, w3), v)
  test2 = w1(v)*wedge_product(w2, w3) - w2(v)*wedge_product(w1, w3) + w3(v)*wedge_product(w1, w2)

  coords1 = test1.get_dense_coordinates()
  coords2 = test2.get_dense_coordinates()
  assert jnp.allclose(coords1, coords2)

################################################################################################################

def differential_form_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs()

  # Create a differential form
  tensor_type = TensorType(0, 3)
  tf = get_tensor_field_fun(M, tensor_type, rng_key)
  w = ParametricDifferentialForm(tf, tensor_type, M)

  # Create some vector fields to evaluate it at
  k1, k2, k3, k4, k5 = random.split(rng_key, 5)
  X = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  Y = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)
  Z = AutonomousVectorField(get_vector_field_fun(M.dimension, k3), M)

  # Try evaluating it at a point and at vector fields
  out1 = w(p)(X(p), Y(p), Z(p))
  out2 = w(X, Y, Z)(p)
  assert jnp.allclose(out1, out2)

  # Try making a wedge product tensor
  tensor_type = TensorType(0, 2)
  tf = get_tensor_field_fun(M, tensor_type, k1)
  nu = ParametricDifferentialForm(tf, tensor_type, M)

  U = AutonomousVectorField(get_vector_field_fun(M.dimension, k4), M)
  V = AutonomousVectorField(get_vector_field_fun(M.dimension, k5), M)
  w_nu = wedge_product_form(w, nu)
  out1 = w_nu(p)(X(p), Y(p), Z(p), U(p), V(p))
  out2 = w_nu(X, Y, Z, U, V)(p)
  assert jnp.allclose(out1, out2)

  # Check the interior product as well
  X_into_w = interior_product_form(w, X)
  out1 = X_into_w(Y, Z)(p)
  out2 = w(X, Y, Z)(p)
  assert jnp.allclose(out1, out2)

################################################################################################################

def example_14p18():
  # Construct the manifolds
  M = EuclideanManifold(dimension=2)
  N = EuclideanManifold(dimension=3)

  # Get a pair of vector fields and a point to use
  rng_key = random.PRNGKey(0)
  k1, k2, k3 = random.split(rng_key, 3)
  X1 = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  X2 = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)
  p = random.normal(k3, (M.dimension,))

  # Construct the map
  def _F(uv):
    u, v = uv
    return jnp.array([u, v, u**2 - v**2])
  F = Map(_F, domain=M, image=N)

  # Get the coordinate functions
  u_mask = jnp.zeros(M.dimension, dtype=bool)
  u_mask = u_mask.at[0].set(True)
  u = M.atlas.charts[0].get_slice_chart(mask=u_mask)

  v_mask = jnp.zeros(M.dimension, dtype=bool)
  v_mask = v_mask.at[1].set(True)
  v = M.atlas.charts[0].get_slice_chart(mask=v_mask)

  x = Map(lambda x: x[0], domain=N, image=EuclideanManifold(dimension=1))
  y = Map(lambda x: x[1], domain=N, image=EuclideanManifold(dimension=1))
  z = Map(lambda x: x[2], domain=N, image=EuclideanManifold(dimension=1))

  # Helper for function differential
  d = lambda f: as_tensor_field(FunctionDifferential(f))

  du = d(u)
  dv = d(v)
  dx = d(x)
  dy = d(y)
  dz = d(z)

  # Construct w
  w = y*(wedge_product_form(dx, dz)) + x*(wedge_product_form(dy, dz))

  # Compute the pullback
  F_star_w = pullback_tensor_field(F, w)

  # Compute the true solution
  ans = -2.0*(u*u + v*v)*wedge_product_form(du, dv)

  F_star_w_p = F_star_w(p)
  ans_p = ans(p)

  coords1 = F_star_w_p.get_dense_coordinates()
  coords2 = ans_p.get_dense_coordinates()

  assert jnp.allclose(coords1, coords2)

################################################################################################################

def exterior_derivative_test():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs(dimension=3)
  k1, k2 = random.split(rng_key, 2)

  # Create a differential form
  tensor_type = TensorType(0, 1)
  tf = get_tensor_field_fun(M, tensor_type, k1)
  w = ParametricDifferentialForm(tf, tensor_type, M)

  tensor_type = TensorType(0, 2)
  tf = get_tensor_field_fun(M, tensor_type, k2)
  n = ParametricDifferentialForm(tf, tensor_type, M)

  # Create some vector fields to evaluate it at
  k1, k2, k3, k4, k5 = random.split(rng_key, 5)
  X = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  Y = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)
  Z = AutonomousVectorField(get_vector_field_fun(M.dimension, k3), M)
  U = AutonomousVectorField(get_vector_field_fun(M.dimension, k4), M)

  # Try making a 0 tensor and using it
  wX = interior_product_form(w, X)
  check = FunctionDifferential(wX)

  test = exterior_derivative(wX)
  out1 = test(Y)

  # Compute the exterior derivative of w
  dw = exterior_derivative(w)
  dn = exterior_derivative(n)

  # Proposition 14.23: Properties of the exterior derivative
  # a)
  c = 0.3
  test1 = exterior_derivative(c*w)
  test2 = c*exterior_derivative(w)
  out1 = test1(X, Y)(p)
  out2 = test2(X, Y)(p)
  assert jnp.allclose(out1, out2)

  # b)
  test1 = exterior_derivative(wedge_product_form(w, n))
  test2 = wedge_product_form(dw, n) + (-1.0)**w.type.l*wedge_product_form(w, dn)
  out1 = test1(X, Y, Z, U)(p)
  out2 = test2(X, Y, Z, U)(p)
  assert jnp.allclose(out1, out2)

  # c)
  ddw = exterior_derivative(exterior_derivative(w))
  out = ddw(X, Y, Z)(p)
  assert jnp.allclose(out, 0.0)

  # d)
  A = random.normal(rng_key, (M.dimension + 1, M.dimension + 1))
  Q, _ = jnp.linalg.qr(A)
  def _F(x, inverse=False):
    if inverse == False:
      return Q@x
    return Q.T@x
  F = Diffeomorphism(_F, domain=M, image=M)

  dF_star_w = exterior_derivative(pullback_differential_form(F, w))
  F_star_dw = pullback_differential_form(F, exterior_derivative(w))

  out1 = dF_star_w(X, Y)(p)
  out2 = F_star_dw(X, Y)(p)

  assert jnp.allclose(out1, out2)

  # Proposition 14.29
  assert w.type.l == 1 # 1 form
  test1 = dw(X, Y)
  test2 = X(w(Y)) - Y(w(X)) - w(lie_bracket(X, Y))
  out1 = test1(p)
  out2 = test2(p)

  assert jnp.allclose(out1, out2)

  # Propostion 14.32
  Xs = [X, Y, Z]
  test1 = dn(*Xs)

  test2 = None
  for i in range(dn.type.l):
    X_no_i = [x for _i, x in enumerate(Xs) if _i != i]
    X_i = Xs[i]
    term = (-1.0)**(i)*X_i(n(*X_no_i))

    if i == 0:
      test2 = term
    else:
      test2 += term

    for j in range(i + 1, dn.type.l):
      X_no_ij = [x for _i, x in enumerate(Xs) if (_i != i) and (_i != j)]
      X_j = Xs[j]
      term = (-1.0)**(i + j)*n(lie_bracket(X_i, X_j), *X_no_ij)
      test2 += term

  out1 = test1(p)
  out2 = test2(p)

  assert jnp.allclose(out1, out2)

################################################################################################################

def structure_coefficient_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs(dimension=3)
  k1, k2 = random.split(rng_key, 2)

  # Proposition 14.30
  frame = AutonomousFrame(get_chart_fun(M.dimension, k1), M)
  basis_vector_fields = frame.to_vector_field_list()
  basis_covector_fields = frame.get_dual_coframe().to_covector_field_list()

  # Get the structure coefficients
  c_matrix = jnp.zeros([M.dimension]*3)
  for j, Ej in enumerate(basis_vector_fields):
    for k, Ek in enumerate(basis_vector_fields):
      if j >= k:
        continue
      Ej_Ek = lie_bracket(Ej, Ek)

      # Get the coordinate functions corresponding to each basis vector
      for i, ei in enumerate(basis_covector_fields):
        cijk = ei(Ej_Ek)
        c_matrix = c_matrix.at[i,j,k].set(cijk(p))

  # Compare against the coordinate functions for dw
  b_matrix = jnp.zeros([M.dimension]*3)
  for i, ei in enumerate(basis_covector_fields):
    dei = exterior_derivative(ei)

    # Get the coordinate functions for each basis vector
    for j, Ej in enumerate(basis_vector_fields):
      for k, Ek in enumerate(basis_vector_fields):
        if j >= k:
          continue

        bijk = dei(Ej, Ek)
        b_matrix = b_matrix.at[i,j,k].set(bijk(p))

  assert jnp.allclose(b_matrix, -c_matrix)

################################################################################################################

def lie_derivative_tests():
  rng_key, M, p, T, T2, v1, v2, v3, v4, v5 = get_inputs(dimension=3)
  k1, k2 = random.split(rng_key, 2)

  # Create a differential form
  tensor_type = TensorType(0, 1)
  tf = get_tensor_field_fun(M, tensor_type, k1)
  w = ParametricDifferentialForm(tf, tensor_type, M)

  tensor_type = TensorType(0, 2)
  tf = get_tensor_field_fun(M, tensor_type, k2)
  n = ParametricDifferentialForm(tf, tensor_type, M)

  # Create some vector fields to evaluate it at
  k1, k2, k3, k4, k5 = random.split(rng_key, 5)
  X = AutonomousVectorField(get_vector_field_fun(M.dimension, k1), M)
  Y = AutonomousVectorField(get_vector_field_fun(M.dimension, k2), M)
  Z = AutonomousVectorField(get_vector_field_fun(M.dimension, k3), M)
  V = AutonomousVectorField(get_vector_field_fun(M.dimension, k4), M)

  # Cartans magic formula
  test1 = lie_derivative(V, n)
  test2 = interior_product_form(exterior_derivative(n), V) + exterior_derivative(interior_product_form(n, V))
  out1 = test1(X, Y)(p)
  out2 = test2(X, Y)(p)
  assert jnp.allclose(out1, out2)

  # Proposition 14.33
  test1 = lie_derivative(V, wedge_product_form(w, n))
  test2 = wedge_product_form(lie_derivative(V, w), n) + wedge_product_form(w, lie_derivative(V, n))
  out1 = test1(X, Y, Z)(p)
  out2 = test2(X, Y, Z)(p)
  assert jnp.allclose(out1, out2)

  # Corollary 14.36
  test1 = lie_derivative(V, exterior_derivative(n))
  test2 = exterior_derivative(lie_derivative(V, n))
  out1 = test1(X, Y, Z)(p)
  out2 = test2(X, Y, Z)(p)
  assert jnp.allclose(out1, out2)

################################################################################################################

def run_all():
  jax.config.update("jax_enable_x64", True)

  alternating_tests()
  elementary_alternating_tensor_tests()
  determinant_tests()
  wedge_product_tests()
  interior_product_tests()
  differential_form_tests()
  example_14p18()
  exterior_derivative_test()
  structure_coefficient_tests()
  lie_derivative_tests()

if __name__ == "__main__":
  from debug import *
  run_all()