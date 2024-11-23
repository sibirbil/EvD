import jax
import jax.numpy as jnp


######################
## SCHEDULERS ET AL ##
######################

def as_scheduler(value):
    """
    Turns scalar into constant step-size function
    """
    if callable(value):
        return value
    return lambda step: value


def power_decay(
    init_lr : jnp.float_,         # the starting learning rate 
    alpha   : jnp.float_,         # decay rate exponent
    offset  : jnp.float_  = 1.,   # in case step count starts from 0
    rate    : int | float = 100   # how many steps  
    ):
    """
    Returns a scheduler which decays by 1/(step/rate + 1)^alpha.
    The rate determines how many steps it takes to 
    """
    def schedule(step: int)-> float:
        return init_lr/ ((step/rate + offset)**alpha)
    
    return schedule

def sqrt_decay(init_lr):
    return power_decay(init_lr, 1/2)

def harmonic_decay(init_lr):
    return power_decay(init_lr, 1)