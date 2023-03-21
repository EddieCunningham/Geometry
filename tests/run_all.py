from tests.lie_group_tests import run_all as run_all_lie_group
from tests.manifold_tests import run_all as run_all_manifold
from tests.map_tests import run_all as run_all_map
from tests.set_tests import run_all as run_all_set
from tests.tangent_tests import run_all as run_all_tangent
from tests.vector_field_tests import run_all as run_all_vector_field
from tests.lie_algebra_tests import run_all as run_all_lie_algebra
from tests.flow_tests import run_all as run_all_flow
from tests.frame_tests import run_all as run_all_frame
from tests.bundle_tests import run_all as run_all_bundle
from tests.cotangent_tests import run_all as run_all_cotangent
import src.util as util
import jax

if __name__ == "__main__":
  from debug import *
  jax.config.update("jax_enable_x64", True)
  # run_all_set()
  # run_all_map()
  # run_all_manifold()
  # run_all_tangent()
  # run_all_lie_group()
  # run_all_vector_field()
  # run_all_lie_algebra()
  # run_all_flow()
  # run_all_bundle()
  run_all_frame()
  # run_all_cotangent()