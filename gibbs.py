import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial

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


def acceptance_ratio(x, x_maybe, xi, hyps):
    func, grad_func, eta = hyps
    w = x - x_maybe + eta*grad_func(x_maybe)
    v = (1/(4*eta))*jnp.inner(w,w)
    u = 1/2* jnp.inner(xi,xi) - func(x_maybe) + func(x) - v
    return jnp.reshape(jnp.exp(jnp.minimum(u,0)),())

def MALA_step(state, hyps):
    func, grad_func, eta = hyps

    def main_func(carry):
        x, key, _, _ = carry
        key, accept_key = random.split(key)
    
        g = grad_func(x)
        x_maybe, xi = langevin_step(x, g, eta, key)
        alpha = acceptance_ratio(x, x_maybe, xi, hyps)
        
        def accept():
            return x_maybe
        
        def reject():
            return x
        
        u = jax.random.uniform(accept_key)
        x_next = jax.lax.cond(u <= alpha, accept, reject)
        return x_next, key, u, alpha
    
    def cond_func(carry):
        _, _, u, alpha = carry
        return u > alpha
        
    init_carry = (state[0], state[1], state[0], 0.0)
    final_carry = jax.lax.while_loop(cond_func, main_func, init_carry) 

    accepted_x, _ ,_ ,_ = final_carry

    return accepted_x, state[1]


def MALA_chain(state, hyps, NSteps):
    def f(carry,_):
        x_next, key = MALA_step(carry, hyps)
        return (x_next, key), x_next
    return jax.lax.scan(f, state, None, length = NSteps)


import matplotlib.pyplot as plt

def plot_trajectory_histogram(trajectory, bins=50):
    plt.figure(figsize=(8, 6))
    
    # Plot histogram of the 1D trajectory
    plt.hist(trajectory, bins=bins, density=True, color='blue', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.title('Histogram of MALA Trajectory')

    # Show the plot
    plt.grid(True)
    plt.show()

#def MALA(xs, func, grad_func, eta, key, nSteps):
    
#jax.lax.scan()

# jnp.exp(3)

# key = random.PRNGKey(0)
# key, _ = random.split(key)
# random.normal(key)