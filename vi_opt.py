import jax
import optax
import jax.numpy as jnp
import jax.random as random
from functools import partial

from typing import NamedTuple, Callable, Tuple, TypeAlias
from jaxtyping import PyTree, Array



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

def gaussian_samples(
    key         : Array, 
    means       : PyTree, 
    stdevs      : PyTree,  # assumed to have the same structure as means
    nSamples    : int = 1
    ):
    """
    Returns a PyTree where jax.Array leaves. Can also be a single jax.Array,
    which is also considered as a PyTree with a single node. 
    Returns a Pytree with tree structure identical to those of means, with 
    jax.Array leaves with the same shape except for one extra leading dimension
    with length = nSamples. 
    This bundles all Gaussian samples into single PyTree. 
    Same standard deviation is used for all entries of all arrays at all leaves. 
    """
    keys = keys_tree(key, means) #different key at each leaf, structured as a tree
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

def gaussian_single_std_samples(key :random.PRNGKey, means :PyTree, stdev : float, nSamples :int = 1):
    """
    same as gaussian_samples, with same standard deviation used for all entries of all leaves.
    """
    stdevs = create_tree(stdev, means)
    return gaussian_samples(key, means, stdevs, nSamples)

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
    noises = gaussian_samples(key, means, stdevs, nSamples)
    return jax.vmap(func)(noises)



@partial(jax.jit, static_argnums=1)
def vi_step(state, hyps: Hyps, step : int):
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
    


