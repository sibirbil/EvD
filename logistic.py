import jax
import jax.numpy as jnp



def logistic_loss(
        xs      :jax.Array,         # data in N x d
        ts      :jax.Array,         # labels given as +1 -1
        thetas  :jax.Array,         # parameters, where last entry is the bias term
        )-> jnp.float_:
    x_appended = _append1(xs) 
    loss = jnp.mean(jax.nn.softplus(-ts*(x_appended @ thetas.transpose()))) 
    return loss

def l1_reg(input: jax.Array):
    """
    l1 norm of the input, which can be a multidimensional array
    """
    return jnp.sum(jnp.absolute(input))

def l2_reg(input: jax.Array):
    """
    l2 norm of the input which can be a multidimensional array
    """
    input_flat = input.flatten()
    return jnp.inner(input_flat, input_flat)

def _append1(x:jax.Array)->jax.Array:
    """
    Returns a matrix with a column of appended 1's
    Input x is either a vector or a matrix
    """
    if len(x.shape)==1:
        x = x[jnp.newaxis,:]
    return jnp.column_stack([x, jnp.ones(len(x))])

def F_function(
        xs          : jax.Array,    # data points
        ts          : jax.Array,    # labels corresponding to data, +/- 1
        reg_type    : str,          # 'l1' or 'l2'
        reg_const   : jnp.float_,   # multiplier in front of regularizer
        beta        : jnp.float_    # inverse temperature
        ):
    reg_fn = l1_reg if reg_type == 'l1' else l2_reg 
    F = lambda theta: beta*(logistic_loss(xs,ts,theta) + reg_const*reg_fn(theta))
    return F, jax.grad(F)

def G_function(
        thetas      : jax.Array,
        t           : int,          # +/- 1
        reg_type    : str,
        reg_const   : jnp.float_,
        beta        : jnp.float_,
        ):
    reg_fn = l1_reg if reg_type == 'l1' else l2_reg
    G = lambda x : beta*(logistic_loss(x, jnp.array([t]), thetas) +  reg_const*reg_fn(x))
    return G, jax.grad(G)



