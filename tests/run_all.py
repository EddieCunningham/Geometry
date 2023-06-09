from tests.lie_group_tests import run_all as run_all_lie_group
from tests.manifold_tests import run_all as run_all_manifold
from tests.map_tests import run_all as run_all_map
from tests.vector_tests import run_all as run_all_vector
from tests.set_tests import run_all as run_all_set
from tests.tangent_tests import run_all as run_all_tangent
from tests.vector_field_tests import run_all as run_all_vector_field
from tests.lie_algebra_tests import run_all as run_all_lie_algebra
from tests.flow_tests import run_all as run_all_flow
from tests.bundle_tests import run_all as run_all_bundle
from tests.cotangent_tests import run_all as run_all_cotangent
from tests.tensor_tests import run_all as run_all_tensor
from tests.lie_derivative_tests import run_all as run_all_lie_derivative
from tests.riemannian_metric_tests import run_all as run_all_riemannian_metric
from tests.differential_form_tests import run_all as run_all_differential_form
from tests.exponential_map_tests import run_all as run_all_exponential_map
from tests.connection_tests import run_all as run_all_connection
import src.util as util
import jax

if __name__ == "__main__":
  from debug import *
  jax.config.update("jax_enable_x64", True)
  run_all_set()
  run_all_vector()
  run_all_map()
  run_all_manifold()
  run_all_tangent()
  run_all_lie_group()
  run_all_vector_field()
  run_all_lie_algebra()
  with util.global_check_off():
    run_all_flow()
  run_all_bundle()
  run_all_cotangent()
  run_all_tensor()
  run_all_lie_derivative()
  run_all_riemannian_metric()
  run_all_differential_form()
  with util.global_check_off():
    run_all_exponential_map()
  run_all_connection()