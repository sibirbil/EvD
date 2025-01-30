import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
import optax
import operator
from typing import Callable



#TRAINING LOGISTIC REGRESSION as 1 layer MLP

class TrainState(train_state.TrainState):
    pass


def create_train_state_from_key(
    key             : jax.Array, 
    model           : nn.Module, 
    input_size      : int,
    optimizer       : optax.GradientTransformation,
    ):

    params = model.init(key, jnp.ones([1,input_size]))
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def create_train_state_from_params(
    params      : dict,
    model       : nn.Module,
    optimizer   : optax.GradientTransformation
    ):
    return TrainState.create(apply_fn= model.apply, params = params, tx = optimizer)

def create_params_from_array2(w: jax.Array, b: jax.Array):
    """
    Given a weight vector w and a bias scalar b for a linear model
    Returns the PyTreee of parameters in the same form as an single layer MLP
    """
    # Ensure weights are in the expected shape for Dense
    w = w.reshape(-1, 1)  # Flatten to (n_features,)
        
    return {'params':{'Dense_0':{'kernel':w, 'bias':b}}}

def create_params_from_array(w: jax.Array, b: jax.Array):
    """
    Given a weight vector w and a bias scalar b for a linear model
    Returns the PyTreee of parameters in the same form as an single layer MLP
    """
    # Ensure weights are in the expected shape for Dense
    w = w.reshape(-1, 1)  # Flatten to (n_features,)
    # Convert bias to a scalar if needed
    if isinstance(b, jnp.ndarray) and b.shape == (1,):
        b = b[0]  # Extract scalar from array
    elif isinstance(b, jnp.ndarray) and b.ndim == 0:
        b = b.item()  # Convert scalar JAX array to Python scalar
        
    return {'params':{'Dense_0':{'kernel':w, 'bias':b}}}
    
def _bias_to_zero_fn(key_path, value):
    """
    takes in the pair (key_path, value) and 
    returns 0 if key_path ends in 'bias', and value otherwise.
    """ 
    return jax.lax.cond(key_path[-1].key=='bias', lambda a : 0., lambda a: a, value)

@jax.jit
def train_step(
        state   : TrainState, 
        X       : jax.Array,
        y       : jax.Array
    )-> TrainState:
    
    def loss_fn(params):
        logits = state.apply_fn(params, X)
        loss = cross_entropy(logits, y)
        regularizer_tree = jax.tree.map(l2_reg, params)        
        regularizer_tree = jax.tree_util.tree_map_with_path(_bias_to_zero_fn, regularizer_tree)
        regularizer = jax.tree.reduce(operator.add, regularizer_tree)
        return loss + regularizer/len(X)

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def test_eval(ts : TrainState, X, y):

    logits = ts.apply_fn(ts.params, X)
    accuracy = jnp.mean((logits>0) == y)
    loss = cross_entropy(logits, y)
    return accuracy, loss

def train(
    ts              : TrainState,
    X               : jax.Array,
    y               : jax.Array,
    nSteps          : int
    )-> TrainState:

    y = jnp.expand_dims(y,1) #one row per example
    # Training loop
    for iStep in range(nSteps):

        ts = train_step(ts, X, y)   

        if iStep%100==0:
            accuracy, loss = test_eval(ts, X, y) 
            print(f"Batches: {iStep},\tTrain Acc: {accuracy:.2%},\tloss: {loss:.4f}")  
    return ts




###  Estimators

def constant_estimator(value):
    return lambda x : value

def logistic_estimator(params, model: nn.Module):
    
    def predictor(x):
        logits = model.apply(params, x)      
        return jnp.round(jax.nn.sigmoid(logits),0)

    return predictor

def negation_logistic_estimator(params, model : nn.Module):

    def predictor(x):
        logits = model.apply(params, x)
        return 1- jnp.round(jax.nn.sigmoid(logits),0)
    return predictor


## Cost functions
def square_cost(logits, y_compare):
    return jnp.mean(jnp.square(jax.nn.sigmoid(logits) - y_compare))

def cross_entropy(logits, y_compare):
    return jnp.mean(jax.nn.softplus(logits) - y_compare*logits)


## Regularizers
def l1_reg(x, C = 1., x0 = 0.):
    return C*jnp.sum(jnp.abs(x - x0))

def l2_reg(x, C=1., x0 = 0.):
    return 0.5*C*jnp.sum(jnp.square(x - x0))




def F_function(
    X       : jax.Array,    # the input variables
    y       : jax.Array,    # the labels
    model   : nn.Module,
    beta    : jnp.float_    # inverse temperature
    ):
    y = jnp.expand_dims(y,1)
    def F(params):
        logits = model.apply(params, X)
        loss = cross_entropy(logits, y)
        regularizer_tree = jax.tree.map(l2_reg, params)
        #bias_to_zero_fn = lambda kp, x: jax.lax.cond(kp[-1].key=='bias', lambda a : 0., lambda a: a, x)
        regularizer_tree = jax.tree_util.tree_map_with_path(_bias_to_zero_fn, regularizer_tree)
        regularizer = jax.tree.reduce(operator.add, regularizer_tree)
        return beta*(loss  + regularizer/len(X))
    return jax.jit(F)

def G_function(
    traj_params     : jax.Array,
    model           : nn.Module,
    estimator_fn    : Callable,  #given x spits out a value y'. Can be constant (read from a list)
    cost_fn         : Callable,
    regularizer_fn  : Callable,
    beta            : float     #inverse temperature
    ):
    
    def G(x):
        logits = jax.vmap(model.apply, in_axes = (0, None))(traj_params, x)[:,:,0]
        loss = cost_fn(logits, estimator_fn(x))
        loss += regularizer_fn(x)
        return beta*loss
    
    return G

