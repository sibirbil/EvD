import jax
import jax.numpy as jnp
import jax.random as random
from jax.tree_util import tree_map, tree_unflatten, tree_structure, tree_leaves, tree_reduce
from functools import partial

def leaf_langevin(
    x       :jax.Array,
    g       :jax.Array,
    xi      :jax.Array,
    eta     :jnp.float_
):
    return x - eta*g + jnp.sqrt(2*eta)*xi



def pytree_langevin_step(
        x,    # current position (PyTree of arrays)
        g,    # gradient of function at position x (PyTree of arrays with same structure as x)
        eta,  # step size (float)
        key   # random number generator key (PRNGKey)
    ):
    """
    Calculates the next step in Langevin dynamics for PyTree inputs.
    It outputs both the new position x_next and the random noise xi added 
    to each element, for use in calculating the acceptance ratio in MALA.
    """

    # Split key to generate unique sub-keys for each leaf in the PyTree
    keys = random.split(key, num = len(tree_leaves(x)))
    keys_tree = tree_unflatten(tree_structure(x), keys)

    # Generate noise for each leaf in the PyTree
    xi = tree_map(lambda k, leaf: random.normal(k, shape=leaf.shape), keys_tree, x)
    
    leaf_langevin_with_eta = partial(leaf_langevin, eta = eta)

    # Apply the single-step update across each leaf in the PyTree
    x_next = tree_map(leaf_langevin_with_eta, x, g, xi)

    return x_next, xi



def langevin_step(
        x:jax.Array,    # current position
        g:jax.Array,    # gradient of function at position x
        eta,            # step size
        key             # random number generator key
        ):
    """
    Calculates the next step in Langevin dynamics.
    It also outputs the random noise added to be used 
    in the calculation of the accceptance ratio for 
    the Metropolis Adjusted Langevin Algorithm (MALA).
    """
    xi = random.normal(key,shape = x.shape)
    x_next = x - eta*g + jnp.sqrt(2*eta)*xi
    return x_next, xi

def ULA_chain(state, hyps, NSteps):
    _, grad_func, eta = hyps    # the function value is not required for ULA

    def f(carry, _):
        x, key = carry
        g = grad_func(x)
        key, subkey = random.split(key)
        x_next, _ = langevin_step(x,g,eta,key)
        return (x_next, subkey), x_next
    
    return jax.lax.scan(f, state, None, length = NSteps)

def F(x):
 	return (x**4)/10 + (x**3)/10 - (x**2)

F_grad = jax.grad(lambda x: jnp.reshape(F(x),()))


hyps0 = (F, F_grad, 0.1)



def MALA_step(state, hyps):
    """
    Computes the next step in Metropolis Adjusted Langevin Algorithm.
    Which either accepts the Langevin step or stays at current point.
    Either case also outputs a new pseudorandom number generator key.
    """
    func, grad_func, eta = hyps
    x, key = state
    
    key, accept_key = random.split(key)
    
    g = grad_func(x)
    x_maybe, xi = langevin_step(x, g, eta, key)
    
    # inlaid function for convenience computes
    def acceptance_ratio():
        w = x - x_maybe + eta*grad_func(x_maybe)
        v = (1/(4*eta)) * jnp.sum(jnp.square(w)) 
        u = (1/(4*eta)) * jnp.sum(jnp.square(xi)) - func(x_maybe) + func(x) - v
        return jnp.reshape(jnp.exp(jnp.minimum(u,0)),())

    # Compute acceptance ratio
    alpha = acceptance_ratio()
    
    # Define acceptance and rejection steps
    def accept():
        return x_maybe

    def reject():
        return x
    
    # Draw uniform random number for the acceptance test
    u = jax.random.uniform(accept_key)
    
    # Decide whether to accept or reject the proposal
    x_next = jax.lax.cond(u <= alpha, accept, reject)

    # Return the accepted (or rejected) state and the updated random key
    return x_next, key



def MALA_chain(state, hyps, NSteps):
    def f(carry,_):
        x_next, key = MALA_step(carry, hyps)
        return (x_next, key), x_next
    return jax.lax.scan(f, state, None, length = NSteps)


def pytree_MALA_step(
        state,
        hyps
    ):
    """
    Performs a MALA step for PyTree inputs. Proposes a new position based on Langevin dynamics 
    and then accepts or rejects it based on the acceptance probability.
    """
    
    x, key = state
    func, grad_func, eta = hyps
    g = grad_func(x)    # a pytree in the same structure as x

    # update key and use the second for acceptance ratio
    key, accept_key = random.split(key)
 
    # Propose a new langevin step
    x_proposed, xi = pytree_langevin_step(x, g, eta, key)

    # compute the gradient at the proposal
    g_proposed = grad_func(x_proposed)

    # Compute acceptance ratio
    def leaf_log_proposal_ratio(x_leaf, x_proposed_leaf, g_proposed_leaf, xi_leaf):
        """Computes the forward-reverse proposal log-probability difference for one leaf."""
        forward = -jnp.sum(jnp.square(xi_leaf)) / (4 * eta)
        reverse = -jnp.sum(jnp.square(x_leaf - x_proposed_leaf + eta * g_proposed_leaf)) / (4 * eta)
        return reverse - forward

    # Sum the proposal log-probability ratios over all leaves
    log_proposal_ratio = tree_reduce(lambda a, b: a + b, tree_map(leaf_log_proposal_ratio, x, x_proposed, g_proposed, xi))
    
    # Compute acceptance probability
    log_acceptance_ratio = -func(x_proposed) + func(x) + log_proposal_ratio
    acceptance_prob = jnp.minimum(1.0, jnp.exp(log_acceptance_ratio))

    # Generate random uniform value to decide acceptance
    uniform_sample = random.uniform(accept_key)
    accepted = uniform_sample < acceptance_prob

    # Choose whether to accept or reject the proposal
    x_next = tree_map(lambda x_p, x_c: jnp.where(accepted, x_p, x_c), x_proposed, x)

    return x_next, key



def pytree_MALA_chain(state, hyps, NSteps):
    def f(carry,_):
        x_next, key = pytree_MALA_step(carry, hyps)
        return (x_next, key), x_next
    return jax.lax.scan(f, state, None, length = NSteps)
