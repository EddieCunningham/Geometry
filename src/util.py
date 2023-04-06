import jax
import jax.tree_util
from typing import Callable, List, Optional, Union, Tuple
from contextlib import contextmanager
import itertools

__all__ = ["global_check_off",
           "tree_shapes",
           "GLOBAL_CHECK",
           "extract_tokens"]

GLOBAL_CHECK = True

@contextmanager
def global_check_off():
  global GLOBAL_CHECK
  GLOBAL_CHECK = False
  try:
    yield
  finally:
    GLOBAL_CHECK = True

def tree_shapes(x):
  return jax.tree_util.tree_map(lambda x: x.shape, x)

################################################################################################################

def extract_tokens(contract):
  """Given a contraction like "k0 l0 l1, k1 l2, k2 k3 l3" return
  the unique tokens in a list like
  ["k0", "l0", "l1", "k1", "l2", "k2", "k3", "l3"] and a function
  called reconstruct that will take as input a list of tokens and
  output a contraction like the original.  For example,
  reconstruct(["l3", "k2", "k3", "l1", "l2", "k0", "l0", "k1"])
  will return "l3 k2 k3, l1 l2, k0 l0 k1"

  Args:
    contract: A contraction for einsum

  Returns:
    A list of the unique tokens in the contraction and a function
    that will reconstruct a similar contraction given a list of tokens.
  """
  # Extract the tokens
  tokens = []
  for elements in contract.split(", "):
    for e in elements.split(" "):
      tokens.append(e)

  def reconstruct(new_tokens):
    assert len(new_tokens) == len(tokens)
    new_tokens_iter = iter(new_tokens)
    new_contract = ""
    for elements in contract.split(", "):
      for e in elements.split(" "):
        new_contract += next(new_tokens_iter) + " "
      new_contract = new_contract[:-1]
      new_contract += ", "
    new_contract = new_contract[:-2]
    return new_contract
  return tokens, reconstruct
