import create_datasets
import nets
import jax
import jax.numpy as jnp

from flax import linen as nn
from typing import Callable, Tuple
import operator

from sklearn.model_selection import train_test_split


adult, _, _  = create_datasets.get_adult()  # check the output signature of the function if it stops working

adult_train, adult_test = train_test_split(adult, test_size = 0.1)

X_train = jnp.array(adult_train.drop(columns = 'income').to_numpy())
y_train = jnp.array(adult_train['income'].to_numpy())

X_test = jnp.array(adult_test.drop(columns = 'income').to_numpy())
y_test = jnp.array(adult_test['income'].to_numpy())



linear = nets.MLP(features=[1], input_size  =  len(adult.columns) - 1)


###  Estimators

def constant_estimator(value):
    return lambda x : value

def logistic_estimator(params, model: nn.Module):
    
    def predictor(x):
        logits = model.apply(params, x)      
        return jnp.round(jax.nn.sigmoid(logits),0)

    return predictor


## Cost functions
def square_cost(logits, y_compare):
    return jnp.mean(jnp.square(jax.nn.sigmoid(logits) - y_compare))

def cross_entropy(logits, y_compare):
    return jnp.mean(jax.nn.softplus(logits) - y_compare*logits)

def logistic_loss(outs, y_compare):
    ts = 2*y_compare - 1
    return jnp.mean(jax.nn.softplus(-ts * outs))

def stupid_logloss(outs, y_compare):
    sigmoids = jax.nn.sigmoid(outs)
    return jnp.mean(-y_compare*jnp.log(sigmoids) - (1- y_compare)*jnp.log(1 - sigmoids))

## Regularizers
def l1_reg(x, C = 1., x0 = 0.):
    return C*jnp.sum(jnp.abs(x - x0))

def l2_reg(x, C=1., x0 = 0.):
    return 0.5*C*jnp.sum(jnp.square(x - x0))




def F_function(
    X       : jax.Array,    # the input variables
    y       : jax.Array,     # the labels
    model   : nn.Module     = linear,
    ):
    def F(params):
        logits = model.apply(params, X)
        loss = cross_entropy(logits, y)
        regularizer_tree = jax.tree.map(l2_reg, params)
        bias_to_zero_fn = lambda kp, x: jax.lax.cond(kp[-1].key=='bias', lambda a : 0., lambda a: a, x)
        regularizer_tree = jax.tree_util.tree_map_with_path(bias_to_zero_fn, regularizer_tree)
        regularizer = jax.tree.reduce(operator.add, regularizer_tree)
        return loss  + regularizer
    return F

def G_function(
    param_traj      : jax.Array,
    model           : nn.Module,
    estimator_fn    : Callable,  #given x spits out a value y'. Can be constant (read from a list)
    cost_fn         : Callable,
    regularizer_fn  : Callable,
    beta            : float     #inverse temperature
    ):
    
    def G(x):
        logits = model.apply(param_traj, x)
        loss = cost_fn(logits, estimator_fn(x))
        loss += regularizer_fn(x)
        return beta*loss
    
    return G


#TRAINING

from flax.training import train_state
import optax


class TrainState(train_state.TrainState):
    pass


def create_train_state(
    key             : jax.Array, 
    model           : nn.Module, 
    input_size      : int,
    optimizer       : optax.GradientTransformation,
    ):

    params = model.init(key, jnp.ones([1,input_size]))
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


from functools import partial
@jax.jit
def train_step(
        state   : TrainState, 
        X       : jax.Array,
        y       : jax.Array
    )-> TrainState:
    
    def loss_fn(params):
        logits = state.apply_fn(params, X)
        #loss = cross_entropy(logits, y)
        #loss = logistic_loss(logits, y)
        #loss = square_cost(logits, y)
        loss = stupid_logloss(logits, X)
        regularizer_tree = jax.tree.map(l2_reg, params)
        bias_to_zero_fn = lambda kp, x: jax.lax.cond(kp[-1].key=='bias', lambda a : 0., lambda a: a, x)
        regularizer_tree = jax.tree_util.tree_map_with_path(bias_to_zero_fn, regularizer_tree)
        regularizer = jax.tree.reduce(operator.add, regularizer_tree)
        return loss + regularizer

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
    key             : jax.random.PRNGKey,
    ts              : TrainState,
    X               : jax.Array,
    y               : jax.Array,
    batch_size      : int,
    nSteps          : int
    )-> TrainState:

    y = jnp.expand_dims(y,1) #one row per example
    # Training loop
    for iStep in range(nSteps):
        # Training
        batch_idx = jax.random.randint(key, batch_size, 0, len(X))
        key, _ = jax.random.split(key)
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        ts = train_step(ts, X_batch, y_batch)   

        if iStep%100==0:
            accuracy, loss = test_eval(ts, X, y) 
            test_accuracy, test_loss = test_eval(ts, X_test, y_test)
            print(f"Batches: {iStep},\tTrain Acc: {accuracy:.2%},\tloss: {loss:.4f}")  
            print(f"\t\t Test Accuracy: {test_accuracy:.2%} \tloss: {test_loss:.4f}")
    return ts