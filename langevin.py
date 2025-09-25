import jax
import jax.numpy as jnp
import jax.random as random
from jax.tree_util import tree_map, tree_unflatten, tree_structure, tree_leaves, tree_reduce
from functools import partial
from typing import Tuple, Optional, List
import torch


# Function for testing 
def F(x):
 	return (x**4)/10 + (x**3)/10 - (x**2)

F_grad = jax.grad(lambda x: jnp.reshape(F(x),()))

hyps = (F, F_grad, 0.1)
state0 = random.PRNGKey(0), 0.   #starting point of 
state1 = random.PRNGKey(1), jnp.array(0.)
state2 = random.PRNGKey(2), jnp.array([0.])


def leaf_langevin(
    x       : jax.Array,
    g       : jax.Array,
    xi      : jax.Array,
    eta     : float,
    clip_to : Tuple[float | None, float | None]
):
    
    step = x - eta*g + jnp.sqrt(2*eta)*xi
    a,b = clip_to
    return jnp.clip(step, min = a, max = b)



def langevin_step(
        key,    # random number generator key (PRNGKey)
        x,      # current position (PyTree of arrays)
        g,      # gradient of function at position x (PyTree of arrays with same structure as x)
        eta,    # step size (float)
        clip_to = [None, None] # projects each step to the preferred interval
    ):
    """
    Calculates the next step in Langevin dynamics for PyTree inputs
    It outputs both the new position x_next and the random noise xi added 
    to each element, for use in calculating the acceptance ratio in MALA
    """

    # Split key to generate unique sub-keys for each leaf in the PyTree
    keys = random.split(key, num = len(tree_leaves(x)))
    keys_tree = tree_unflatten(tree_structure(x), keys)

    # Generate noise for each leaf in the PyTree
    xi = tree_map(lambda k, leaf: random.normal(k, shape=leaf.shape), keys_tree, x)
    
    leaf_langevin_with_args = partial(leaf_langevin, eta = eta, clip_to=clip_to)

    # Apply the single-step update across each leaf in the PyTree
    x_next = tree_map(leaf_langevin_with_args, x, g, xi)

    return x_next, xi


def MALA_step(
        state,
        hyps
    ):
    """
    Performs a MALA step for PyTree inputs. Proposes a new position based on Langevin dynamics 
    and then accepts or rejects it based on the acceptance probability
    """
    
    key, x = state
    #deals with cases when no clip_to value is provided or provided as None
    func, grad_func, eta, *clip_to = (*hyps, None, None)[:5]
    g = grad_func(x)    # a pytree in the same structure as x

    # update key and use the second for acceptance ratio
    key, accept_key = random.split(key)
 
    # Propose a new langevin step
    x_proposed, xi = langevin_step(key, x, g, eta, clip_to)

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

    return key, x_next


from utils import as_scheduler

def MALA_chain(state, hyps, NSteps):
    func, grad_func, eta, *clip_to = (*hyps, None, None)[:5]
    eta = as_scheduler(eta) 
    carry_init = *state, 0

    def f(carry, _):
        key, x, step = carry
        lr = eta(step)
        new_hyps = (func, grad_func, lr, *clip_to)
        
        next_key, x_next = MALA_step((key, x), new_hyps)
        return (next_key, x_next, step + 1), x_next

    (last_key, last_x, _), x_traj = jax.lax.scan(f, carry_init, None, length=NSteps)
    return (last_key, last_x), x_traj


def nt_MALA(state, hyps, Nsteps):
    func, grad_func, eta, *clip_to = (*hyps, None, None)[:5]
    eta = as_scheduler(eta)
    key, x = state
    
    output = []
    for iStep in range(Nsteps):
        lr = eta(iStep)
        new_hyps = (func,grad_func, lr, *clip_to)
        key, x = MALA_step((key, x), new_hyps) 
        output.append(x)

    
    return (key, x), jax.tree.map(lambda *xs: jnp.stack(xs), *output)
     
##################################
## TORCH.TENSOR version of MALA ##
##################################

def torch_langevin_step(
        x: torch.Tensor, 
        g: torch.Tensor, 
        eta: float, 
        clip_to: List[Optional[float]] = [None, None]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the next step in Langevin dynamics for a single tensor.
    Returns both the new position x_next and the random noise xi.
    """
    # Generate noise with same shape as x
    xi = torch.randn_like(x)
    
    # Apply Langevin update: x - eta*g + sqrt(2*eta)*xi
    x_next = x - eta * g + torch.sqrt(torch.tensor(2 * eta, device=x.device)) * xi
    
    # Apply clipping if specified
    a, b = clip_to
    if a is not None or b is not None:
        x_next = torch.clamp(x_next, min=a, max=b)
    
    return x_next.detach(), xi


def torch_MALA_step(x: torch.Tensor, hyps) -> torch.Tensor:
    """
    Performs a MALA step for a single tensor with Metropolis-Hastings acceptance.
    """
    func, grad_func, eta, *clip_to = (*hyps, None, None)[:5]
    
    # Compute gradient at current position
    g = grad_func(x)
    
    # Propose new step using Langevin dynamics
    x_proposed, xi = torch_langevin_step(x, g, eta, clip_to)
    
    # Compute gradient at proposed position
    g_proposed = grad_func(x_proposed.requires_grad_())
    
    # Compute log proposal ratio
    forward = -torch.sum(xi**2) / (4 * eta)
    reverse = -torch.sum((x - x_proposed + eta * g_proposed)**2) / (4 * eta)
    log_proposal_ratio = reverse - forward
    
    # Compute acceptance probability
    log_acceptance_ratio = -func(x_proposed) + func(x) + log_proposal_ratio
    acceptance_prob = torch.minimum(torch.tensor(1.0), torch.exp(log_acceptance_ratio))
    
    # Generate random uniform value for acceptance decision
    uniform_sample = torch.rand(1).item()
    accepted = uniform_sample < acceptance_prob.item()
    
    # Accept or reject proposal
    x_next = x_proposed if accepted else x
    
    return x_next

def torch_MALA_chain(x: torch.Tensor, hyps, NSteps: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    PyTorch implementation of MALA chain for single tensors.
    """
    func, grad_func, eta, *clip_to = (*hyps, None, None)[:5]
    eta = as_scheduler(eta)
    
    x.requires_grad_(True)
    trajectory = []
    
    for step in range(NSteps):
        lr = eta(step)
        new_hyps = (func, grad_func, lr, *clip_to)
        x = torch_MALA_step(x, new_hyps)
        trajectory.append(x.detach())
    
    return x, torch.stack(trajectory)

####################################################
### numpy implementation for locally const funcs ###
####################################################

import numpy as np
from typing import NamedTuple, Callable, Union

Array = np.ndarray

class Hyperparameters(NamedTuple):
    func            : Callable[[Array], Array]  
    # func is a scalar valued function to apply MALA to,
    # with signature will be (n, *x.shape)-> (n,) (i.e. batchable)
    beta            : float                     # inverse temperature
    eta             : float                     # initial step size
    # if grad is not given we will approximate 
    # the gradient of gaussian smoothed func.
    grad_func       : Optional[Callable]  = None 
    reg             : Optional[Callable]  = None
    # target_rate alters step size so that the MALA acceptance rate is within 
    # 0.8 and 1.2 of the target_rate. If target_rate==None then constant step size
    target_rate     : Optional[float] = 0.574    
    # target_change alters smoothing rate so that MALA changes leaf within 
    # 0.8 and 1.2 percent of target_change. If None then smoothing is constant
    target_change   : Optional[float] = 0.1   
    check_every     : int = 100
    n_samples       : Optional[int] = 1         # number of samples to take for grad comp
    sigma           : Optional[float] = 0.1     # sigma of the noise
    

def convolve_and_diff(
        x           : Array,
        n_samples   : int,
        sigma       : float,
        func        : Callable,
        grad_func   : Optional[Callable]
    ):
    """
    Get an estimate for the derivative of E[func], the expectation of func,
    with respect to the mean of the distribution. The distribution is a 
    normal distribution with fixed variance sigma^2. This is equal to the
    correlation of F and gaussian noise itself, i.e. ∫F(x + ε)ε dε, which
    is estimated by random Gaussian samples, making sure -/+ ε are included
    (antithetic sampling), to reduce variance in the estimation.
    """
    x_size = x[0].shape
    if grad_func is None:
        noise_base = sigma * np.random.randn(n_samples, *x_size)
        x_pos = x + noise_base    # (n_samples, *x_size)
        x_neg = x - noise_base    # (n_samples, *x_size)

        T_pos = func(x_pos)
        T_neg = func(x_neg)
        # Below approximates the gradient of ∫F(x + σε)ε dε for F = func
        # where ε ~ N(0,I). Better behaved than ∇F when F is a step function.
        g = (T_pos - T_neg) @ noise_base/(2*sigma*n_samples)
        g = g[None,:]

    else:
        g = grad_func(x)

    return g

def np_MALA_step(
        x       : Array, 
        hyps    : Hyperparameters, 
        eta     : Optional[float] = None,
        sigma   : Optional[float] = None
    ):

    if eta is None:
        eta = hyps.eta/hyps.beta
    if sigma is None:
        sigma = hyps.sigma
    
    g = convolve_and_diff(x, hyps.n_samples, sigma, hyps.func, hyps.grad_func)
    g *= hyps.beta

    # Generate noise with same shape as x
    xi = np.random.randn(*x.shape)
    
    # Apply Langevin update: x - eta*g + sqrt(2*eta)*xi
    update = -eta*g + np.sqrt(2*eta)*xi
    x_proposed = x + update
    # Acceptance probability assuming ∇F(x) = ∇F(x') which may be the case 
    # when ∇F is the gradient of the gaussian convolution of a step function, 
    # since the convolution is flat within leaves or mostly linear in transitions. 
    # log q(x|x') - log q(x'|x) = (∇F)^T (-η(∇F) + sqrt(2η)ξ) which is computed
    # by noting that x' = x - η(∇F(x)) + sqrt(2η)ξ and to go back we would need
    # x = x' - η(∇F(x')) + sqrt(2η)ξ', for a particular ξ'. Which ends up equaling 
    # ξ' = sqrt(2η)∇F(x) - ξ. Then we can calculate the log transition density as 
    # q(x|x') = -(1/2)||ξ'||^2. The forward transition log density is similarly 
    # q(x'|x) = -(1/2)||ξ||^2. Subtracting we get the cancellation and the result.
    log_acc_ratio = -hyps.beta*hyps.func(x_proposed) + hyps.beta*hyps.func(x) + np.dot(g, update[0]) 
    if hyps.reg is not None:
        log_acc_ratio += -hyps.reg(x_proposed) + hyps.reg(x)              
    acc_prob = np.exp(np.minimum(0., log_acc_ratio))
    
    # Generate random uniform value for acceptance decision
    uniform_sample = np.random.rand()
    accepted = (uniform_sample < acc_prob).squeeze()
    
    # Accept or reject proposal
    x_next = x_proposed if accepted else x
    
    changed = (hyps.func(x_next) != hyps.func(x))
    changed = changed.squeeze()
    
    return x_next, accepted, changed


def adaptive_np_MALA(
    x       : Array,
    hyps    : Hyperparameters,
    Nsteps  : int
    ):
    acc = 0     # number of accepted steps
    chg = 0     # number of steps where leaf changed
    tot = 0     # total steps taken
    eta = hyps.eta/hyps.beta
    sigma = hyps.sigma
    traj = [x.squeeze()]
    for step in range(Nsteps):
        x, accepted, changed = np_MALA_step(x, hyps, eta, sigma)
        acc += accepted     # accepted is boolean and is converted to 0 or 1.
        chg += changed      # changed is also a boolean
        tot += 1
        traj.append(x.squeeze())
        if step % hyps.check_every == hyps.check_every-1:
            eta, sigma = adapt_parameters(acc, chg, tot, eta, sigma, hyps)
            print(f"On step {step} we have η={eta*hyps.beta:.6f} and σ={sigma:.4f}")
            acc = 0
            chg = 0
            tot = 0     #reset the counters

            
    return x, np.array(traj)

def adapt_parameters(acc, chg, tot, eta, sigma, hyps: Hyperparameters):

    if hyps.target_rate is None:
        pass        # eta stays fixed
    elif acc/tot > hyps.target_rate*1.2:
        eta = min(1.1*eta, 1./hyps.beta)
    elif acc/tot < hyps.target_rate*0.8:
        eta = max(0.9*eta, 1e-3/hyps.beta)
            
    if hyps.target_change is None:
        pass        # sigma stays fixed
    elif chg/tot > hyps.target_change*2:
        sigma = max(0.9*sigma, 1e-4)   # smaller sigma decreases horizon
    elif chg/tot < hyps.target_change/2:
        sigma = min(1.1*sigma, 1.)      # higher sigma increases horizon
    
    print(f"In the last {hyps.check_every} steps {acc} accepted, {chg} times leaf changed")

    return eta, sigma
            
######################################################
# Below is an earlier version which only works with  #
# jax.Array's and not PyTree's. Parameters of neural #
# networks are structured as PyTrees                 #
######################################################

# def langevin_step(
#     key,                   # random number generator key
#     x       :jax.Array,    # current position
#     g       :jax.Array,    # gradient of function at position x
#     eta,                   # step size
#     ):
#     """
#     Calculates the next step in Langevin dynamics.
#     It also outputs the random noise added to be used 
#     in the calculation of the accceptance ratio for 
#     the Metropolis Adjusted Langevin Algorithm (MALA).
#     """
#     xi = random.normal(key,shape = x.shape)
#     x_next = x - eta*g + jnp.sqrt(2*eta)*xi
#     return x_next, xi



# def ULA_chain(state, hyps, NSteps):
#     _, grad_func, eta = hyps    # the function value is not required for ULA

#     def f(carry, _):
#         key, x = carry
#         g = grad_func(x)
#         key, subkey = random.split(key)
#         x_next, _ = langevin_step(key, x, g, eta)
#         return (subkey, x_next), x_next
    
#     return jax.lax.scan(f, state, None, length = NSteps)



# def MALA_step(state, hyps):
#     """
#     Computes the next step in Metropolis Adjusted Langevin Algorithm.
#     Which either accepts the Langevin step or stays at current point.
#     Either case also outputs a new pseudorandom number generator key.
#     """
#     key, x = state
#     func, grad_func, eta = hyps
    
#     key, accept_key = random.split(key)
    
#     g = grad_func(x)
#     x_maybe, xi = langevin_step(key, x, g, eta)
    
#     # inlaid function for convenience computes
#     def acceptance_ratio():
#         w = x - x_maybe + eta*grad_func(x_maybe)
#         v = (1/(4*eta)) * jnp.sum(jnp.square(w)) 
#         u = (1/(4*eta)) * jnp.sum(jnp.square(xi)) - func(x_maybe) + func(x) - v
#         return jnp.reshape(jnp.exp(jnp.minimum(u,0)),())

#     # Compute acceptance ratio
#     alpha = acceptance_ratio()
    
#     # Define acceptance and rejection steps
#     def accept():
#         return x_maybe

#     def reject():
#         return x
    
#     # Draw uniform random number for the acceptance test
#     u = jax.random.uniform(accept_key)
    
#     # Decide whether to accept or reject the proposal
#     x_next = jax.lax.cond(u <= alpha, accept, reject)

#     # Return the accepted (or rejected) state and the updated random key
#     return key, x_next



# def MALA_chain(state, hyps, NSteps):
#     def f(carry,_):
#         key, x_next = MALA_step(carry, hyps)
#         return (key, x_next), x_next
#     return jax.lax.scan(f, state, None, length = NSteps)