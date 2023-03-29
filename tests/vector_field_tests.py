from functools import partial
from typing import Callable
from typing import NewType
import src.util
import jax
import jax.numpy as jnp
import jax.random as random
from src.set import *
from src.map import *
from src.tangent import *
from src.manifold import *
from src.vector_field import *
from src.lie_derivative import *
from src.instances.manifolds import *
from src.instances.vector_fields import *
import src.util as util

################################################################################################################

def get_vector_field_fun(dimension, rng_key):
  # Construct the vector field over M
  import nux
  net = nux.CouplingResNet1D(out_dim=dimension,
                             working_dim=8,
                             hidden_dim=16,
                             nonlinearity=nux.util.square_swish,
                             dropout_prob=0.0,
                             n_layers=1)
  x = random.normal(rng_key, (100, dimension))
  net(x, rng_key=rng_key)
  params = net.get_params()
  leaves, treedef = jax.tree_util.tree_flatten(params)
  keys = random.split(rng_key, len(leaves))
  new_leaves = [random.normal(key, x.shape) for x, key in zip(leaves, keys)]
  params = treedef.unflatten(new_leaves)

  def vf(x):
    return net(x[None], params=params, rng_key=rng_key)[0]

  return vf

################################################################################################################

def run_all():
  import nux
  # jax.config.update('jax_disable_jit', True)
  jax.config.update("jax_enable_x64", True)
  rng_key = random.PRNGKey(0)

  # Construct a manifold
  M = Sphere(dim=4)
  p = random.normal(rng_key, (5,)); p = p/jnp.linalg.norm(p)

  # Construct 2 different vector fields
  key1, key2, key3 = random.split(rng_key, 3)
  X = AutonomousVectorField(get_vector_field_fun(M.dimension, key1), M)
  Y = AutonomousVectorField(get_vector_field_fun(M.dimension, key2), M)
  Z = AutonomousVectorField(get_vector_field_fun(M.dimension, key3), M)

  f = Map(lambda x: jnp.cos(4*x**2 + jnp.sin(x)).sum(), domain=M, image=Reals(dimension=1))
  g = Map(lambda x: jnp.sin(2*x**2 + jnp.cos(x)).sum(), domain=M, image=Reals(dimension=1))

  def on_sphere(x, inverse=False):
    if inverse == False:
      rphi = nux.CartesianToSpherical()(x[None])[0]
      r, phi = rphi[...,:1], rphi[...,1:]
      phi = phi + 0.1
      rphi = jnp.concatenate([r, phi], axis=-1)
      return nux.SphericalToCartesian()(rphi)[0][0]
    else:
      rphi = nux.CartesianToSpherical()(x[None])[0]
      r, phi = rphi[...,:1], rphi[...,1:]
      phi = phi - 0.1
      rphi = jnp.concatenate([r, phi], axis=-1)
      return nux.SphericalToCartesian()(rphi)[0][0]

  F = Diffeomorphism(on_sphere, domain=M, image=M)
  q = F(p)
  assert jnp.allclose(F.inverse(q), p)

  # Test that we can evaluate the vector field at a point
  Xp = X(p)
  Yp = Y(p)

  Xpf = Xp(f)
  Ypf = Yp(f)

  # Test that we can add the vector fields
  assert jnp.allclose((X + Y)(p)(f), Xpf + Ypf)
  assert jnp.allclose((Y + X)(p)(f), Xpf + Ypf)

  # Test that we can multiply by scalars
  a, b = random.normal(rng_key, (2,))
  assert jnp.allclose((a*X)(p)(f), a*Xpf)
  assert jnp.allclose((f*X)(p)(f), f(p)*(Xpf))
  assert jnp.allclose((f*X + g*Y)(p)(f), f(p)*(Xpf) + g(p)*Y(p)(f))

  # Test left hand side multiplication
  assert jnp.allclose((X(f))(p), X(p)(f))

  # Test the product rule
  fg = Map(lambda x: f(x)*g(x), domain=M, image=Reals(dimension=1))
  Xfg = X(fg)
  out_Xfg = Xfg(p)

  fXg = (f*X)(g)
  out_fXg = fXg(p)

  gXf = (g*X)(f)
  out_gXf = gXf(p)
  assert jnp.allclose(out_Xfg, out_fXg + out_gXf)

  # Check the pushforward
  FX = pushforward(F, X)
  FXf = FX(f)

  lhs = compose(pushforward(F, X)(f), F)(p)
  rhs = (X(compose(f, F)))(p)
  assert jnp.allclose(lhs, rhs)


  # Check the Lie bracket of vector fields
  X_Y = lie_bracket(X, Y)
  X_Yp = X_Y(p)

  lhs = X_Yp(f)
  rhs = X(p)(Y(f)) - Y(p)(X(f))
  assert jnp.allclose(lhs, rhs)

  # Check its identities:
  # Bilinearity
  lhs = lie_bracket(a*X + b*Y, Z)(p)(f)
  rhs = (a*lie_bracket(X, Z) + b*lie_bracket(Y, Z))(p)(f)
  assert jnp.allclose(lhs, rhs)

  # Antisymmetry
  lhs = lie_bracket(X, Y)(p)(f)
  rhs = -lie_bracket(Y, X)(p)(f)
  assert jnp.allclose(lhs, rhs)

  # Jacobi Identity
  t1 = lie_bracket(X, lie_bracket(Y, Z))
  t2 = lie_bracket(Y, lie_bracket(Z, X))
  t3 = lie_bracket(Z, lie_bracket(X, Y))
  check = (t1 + t2 + t3)(p)(f)
  assert jnp.allclose(check, 0.0)

  # Product rule
  lhs = lie_bracket(f*X, g*Y)(p)(f)
  rhs = (fg*lie_bracket(X, Y) + ((f*X)(g))*Y - ((g*Y)(f))*X)(p)(f)
  assert jnp.allclose(lhs, rhs)

  # Pushforward
  rhs = pushforward(F, lie_bracket(X, Y))(p)(f)
  lhs = lie_bracket(pushforward(F, X), pushforward(F, Y))(p)(f)
  assert jnp.allclose(lhs, rhs)

if __name__ == "__main__":
  from debug import *
  jax.config.update("jax_enable_x64", True)
  run_all()