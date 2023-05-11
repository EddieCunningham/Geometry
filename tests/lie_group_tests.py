from functools import partial
from typing import Callable, Optional
import src.util
import jax.random as random
import jax.numpy as jnp
from src.set import *
from src.map import *
from src.manifold import *
from src.tangent import *
from src.lie_group import *
from src.instances.manifolds import *
from src.instances.lie_groups import *
import src.util as util

def general_linear_test(group, g, h, l, M, p):
  assert (g in group) and (h in group) and (l in group)

  # Check the different operations
  assert jnp.allclose(group.inverse(g), jnp.linalg.inv(g))
  assert jnp.allclose(group.get_identity_element(), jnp.eye(group.N))
  assert jnp.allclose(group.multiplication_map(g, h), g@h)
  assert jnp.allclose(group.left_translation_map(g)(h), g@h)
  assert jnp.allclose(group.right_translation_map(g)(h), h@g)

  # Show that the conjugate is a Lie group homomorphism
  C_l = group.conjugation_map(l)
  assert jnp.allclose(C_l(group.multiplication_map(g, h)), group.multiplication_map(C_l(g), C_l(h)))

  # Confirm that conjugate map has constant rank
  C_g = group.conjugation_map(l)
  J1 = Differential(C_l, l).get_coordinates()
  J2 = Differential(C_g, g).get_coordinates()
  assert jnp.linalg.matrix_rank(J1) == jnp.linalg.matrix_rank(J2)

  # Check that the translation maps on a manifold compose as expected
  theta_g = group.left_action_map(g, M)
  theta_h = group.left_action_map(h, M)
  theta_hg = group.left_action_map(group.multiplication_map(h, g), M)
  g_p = theta_g(p)
  hg_p1 = theta_h(g_p)
  hg_p2 = theta_hg(p)
  assert jnp.allclose(hg_p1, hg_p2)

  theta_g = group.right_action_map(g, M)
  theta_h = group.right_action_map(h, M)
  theta_hg = group.right_action_map(group.multiplication_map(h, g), M)
  h_p = theta_h(p)
  hg_p1 = theta_g(h_p)
  hg_p2 = theta_hg(p)
  assert jnp.allclose(hg_p1, hg_p2)

def internal_semidirect_product_test(rng_key):

  # Construct the internal semi-direct product of N and H
  # In this case, we'll have that the map (n,h) -> nh is a Lie group isomorphism between isp(N, H) and G.
  dim = 4
  G = GLRn(dim=dim)
  N = GLp(dim=dim)
  H = GLp(dim=dim)
  NH = internal_semidirect_product(N, H)

  # This is a Lie group isomorphism between isp(N, H) and G
  def _rho(g):
    n, h = g
    return G.multiplication_map(n, h)
  rho = Map(_rho, domain=NH, image=G)

  g, n, h = random.normal(rng_key, (3, G.N, G.N))
  n, h = n.T@n, h.T@h

  # Look at its differential
  drho = Differential(rho, (n, h))
  J = drho.get_coordinates()

  # assert jnp.allclose(J[:,dim**2:], 0.0)

################################################################################################################

def run_all():
  import jax
  jax.config.update("jax_enable_x64", True)

  rng_key = random.PRNGKey(0)
  M = GLRn(dim=4)
  p = random.normal(rng_key, (M.N, M.N))

  group = GLRn(dim=4)
  g, h, l = random.normal(rng_key, (3, group.N, group.N))
  general_linear_test(group, g, h, l, M, p)

  group = GLp(dim=4)
  g, h, l = g.T@g, h.T@h, l.T@l
  general_linear_test(group, g, h, l, M, p)

  # Ensure that the semi-direct product is built using an action by automorphisms
  NH = semidirect_product(group, group)
  x = (g, h)
  x_inv = NH.inverse(x)
  assert jax.tree_util.tree_map(jnp.allclose, *(NH.get_identity_element(), NH.multiplication_map(x, x_inv)))

  # Check that the Euclidean group works
  group = EuclideanGroup(dim=4)
  b, A = jnp.ones(4), jnp.linalg.qr(g)[0]
  x = (b, A)
  x_inv = group.inverse(x)
  assert jax.tree_util.tree_map(jnp.allclose, *(group.get_identity_element(), group.multiplication_map(x, x_inv)))

  theta_g = group.left_action_map(x, RealLieGroup(dimension=4))
  theta_g(jnp.ones(4)*2)

  # Test the internal_semidirect_product
  internal_semidirect_product_test(rng_key)

if __name__ == "__main__":
  from debug import *
  run_all()