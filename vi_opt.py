import jax
import optax
import jax.numpy as jnp
import jax.random as random
from create_datasets import get_MNIST
from datasets import Dataset
from nets import MLP, cross_entropy_loss
from functools import partial

from collections import namedtuple
from typing import NamedTuple, Callable


key = random.PRNGKey(0)

#hyps = namedtuple('Hyps', ['f', 'grad_f', 'learning_rate', 'nSamples', 'base_dist'])
Hyps = NamedTuple(
    'Hyps',
    [
        ('func', Callable),         # function to be optimized
        ('grad_func', Callable),    # grad of the function to be optimized
        ('lr', Callable),           # learning rate as a (possibly constant) schedule
        ('nSamples', int),          # number of samples in VI ensemble
        ('scale', float)            # scaling factor for the base distribution
    ]
    )

def create_tree(x, tree):
    """
    Creates a DictTree of the same structure and 
    node names as tree but with leaf values x.
    """
    return jax.tree.map(lambda _: x, tree)

def keys_tree(key, tree):
    """
    Creates a PRNGKey tree, structured as tree
    """
    treedef = jax.tree_structure(tree)
    keys = random.split(key, treedef.num_leaves)
    keys_tree = jax.tree_unflatten(treedef, keys)
    return keys_tree

def gaussian_samples_tree(key, means, stdev, nSamples = 1):
    """
    Returns a PyTree where jax.Array leaves. The shape of arrays at leaves
    is identical to those of means, except for one extra leading dimension
    with length = nSamples. This bundles all Gaussian samples into single PyTree.
    stdev is expected to be a PyTree with same structure as means, but can 
    be a scalar, in which case same same standard deviation is used for all
    entries of all arrays at all leaves. 
    """
    stdevs = create_tree(stdev, means)
    keys = keys_tree(key, means) #structured as a tree
    noises = jax.tree.map(
        lambda l, k: jax.random.normal(k, (nSamples, *l.shape), l.dtype), 
        means, 
        keys
        )
    return jax.tree.map(
        lambda m, s, n: m + n*s,
        means,
        stdevs,
        noises
    ) 

@partial(jax.jit, static_argnames = ('func', 'nSamples'))
def noisy_vals(
    key,    # random number generator key
    func,   # a function to be evaluated (thought of as gradient of loss) 
    means,  # mean of input variable samples
    stdevs, # standard deviation of input variable samples
    nSamples = 1 # number of samples. 
    ):
    """
    Applies func to samples from Gaussian distribution.
    nSample many different parameters are constructed with a
    distribution coming from a gaussian with a given mean and std.
    These are thought to be parameters of a neural network.
    Then func (which is thought of as gradient of a loss function)
    is evaluated at these samples (to produce gradient vectors)
    
    Returns an array of outputs of func with leading dimension of size nSamples.
    """
    noises = gaussian_samples_tree(key, means, stdevs, nSamples)
    return jax.vmap(func)(noises)



@partial(jax.jit, static_argnums=1)
def vi_step(state, hyps: Hyps, step):
    key, x0= state
    key, subkey = random.split(key)
    grads = noisy_vals(key, hyps.grad_func, x0, hyps.scale, hyps.nSamples)
    av_grad = jnp.mean(grads, axis = 0)
    x_next = jax.tree.map(lambda x, g: x + hyps.lr(step)*g, x0, av_grad)
    return subkey, x_next


def vi_traj(state, hyps: Hyps, Nsteps):
    def f(carry, step):
        new_carry = vi_step(carry, hyps, step)
        return new_carry, new_carry[1]
    _, traj = jax.lax.scan(f, state, jnp.arange(Nsteps), Nsteps)
    return traj

@partial(jax.jit, static_argnums = 1)
def vi_train(state, hyps, Nsteps):
    def f(step, carry):
        new_carry = vi_step(carry, hyps, step)
        return new_carry
    return jax.lax.fori_loop(0, Nsteps, f, state)
    


