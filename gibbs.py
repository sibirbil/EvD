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
    xi = random.normal(key,shape = x.shape)
    x_next = x - eta*g + jnp.sqrt(2*eta)*xi
    return x_next, xi

def ULA_chain(state, hyps, NSteps):
    x, key = state
    _, grad_func, eta = hyps
    g = grad_func(x)

    def f(carry, _):
        x, key = carry
        key, subkey = random.split(key)
        x_next, _ = langevin_step(x,g,eta,key)
        return (x_next, key), x_next
    
    init_carry = (x,key)
    return jax.lax.scan(f, init_carry, None, length = NSteps)

def F(x):
 	return (x**4)/10 + (x**3)/10 - (x**2)

F_grad = jax.grad(F)


hyps0 = (F, F_grad, 0.1)


def acceptance_ratio(x, x_maybe, xi, hyps):
    func, grad_func, eta = hyps
    w = x - x_maybe + eta*grad_func(x_maybe)
    v = (1/(4*eta))*jnp.inner(w,w)
    u = 1/2* jnp.inner(xi,xi) - func(x_maybe) + func(x) - v
    return jnp.exp(jnp.minimum(u,0))

def MALA_step(state, hyps):
    func, grad_func, eta = hyps
    x, key = state

    def main_func(carry):
        x, key, _, _ = carry
        key, accept_key = random.split(key)
    
        g = grad_func(x)
        x_maybe, xi = langevin_step(x, g, eta, key)
        alpha = acceptance_ratio(x, x_maybe, xi, hyps)
        
        def accept(carry):
            return x_maybe, key
        
        def reject(carry):
            return x, key
        
        u = jax.random.uniform(accept_key)
        x_next, _ = jax.lax.cond(u <= alpha, accept, reject, carry)
        return x_next, accept_key, x_maybe, alpha
    
    def cond_func(carry):
        _, accept_key, _, alpha = carry
        u = jax.random.uniform(accept_key)
        return u > alpha
        
    init_carry = (x, key, x, 0.0)
    final_carry = jax.lax.while_loop(cond_func, main_func, init_carry) 

    accepted_x, _ ,_ ,_ = final_carry

    return accepted_x, key


def MALA_chain(state, hyps, NSteps):
    def f(carry,_):
        x_next, key = MALA_step(carry, hyps)
        return (x_next, key), x_next
    return jax.lax.scan(f, state, None, NSteps)


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