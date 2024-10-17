import jax
import jax.numpy as jnp
import jax.random as random

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
        v = (1/(4*eta))*jnp.inner(w,w)
        u = 1/2* jnp.inner(xi,xi) - func(x_maybe) + func(x) - v
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



