from functools import partial
from typing import Callable, List, Optional
import src.util
from functools import partial
import jax
import jax.numpy as jnp
from src.set import *
from src.manifold import *
from src.map import *
from src.tangent import *
from src.instances.manifolds import Vector, VectorSpace
import src.util as util

class SetOfMultilinearMaps(Map[List[Vector],Vector]):
  pass

def tensor_product(w: Tensor, n: Tensor) -> Tensor:
  """Tensor product

  Args:
    w: Tensor 1
    n: Tensor 2

  Returns:
    A new tensor of degree equal to the sum of that of w and n
  """
  pass