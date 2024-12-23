import jax
import jax.numpy as jnp
import jax.random as random

from flax import linen as nn
from flax.training import train_state

from datasets import Dataset

import optax

from typing import Callable, Sequence

###################
##### LOSSES ######
###################


# for index valued labels, no ensemble dimension
def cross_entropy_loss(batch_logits, batch_labels):
    """
    Returns cross entropy loss for inputs of shape (B, K)
    and labels are indices in [0..(K-1)] of shape (B,)
    Mean is over the batch dimension.
    """
    logprobs = jax.nn.log_softmax(batch_logits)
    @jax.vmap # for each batch member gets the label column of the B x K logits matrix.
    def index(logits, label):
        return logits[label]
    return -jnp.mean(index(logprobs,batch_labels))

def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels, axis =-1)
    
###################
#### TRAINING #####
###################

class TrainState(train_state.TrainState):
    rng_key : jax.Array #to be used as seed for a dropout layer, in case it exists


def create_train_state(
    key             : jax.Array, 
    model           : nn.Module, 
    input_shape     : Sequence[int], 
    optimizer       : optax.GradientTransformation,
    ):
    param_key, dropout_key = random.split(key)
    batch_x = jnp.ones([1, *input_shape])
    params = model.init(param_key, batch_x)

    return TrainState.create(apply_fn=model.apply, params=params, rng_key = dropout_key, tx=optimizer)


@jax.jit
def train_step(
        state   : TrainState, 
        batch   : dict,
    )-> TrainState:

    dropout_train_key = jax.random.fold_in(key=state.rng_key, data=state.step)
    def loss_fn(params):
        logits = state.apply_fn(params, batch['x'], rngs = {'dropout': dropout_train_key})
        loss = cross_entropy_loss(logits, batch['y'])
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(ts : TrainState, batch : Dataset):

    logits = ts.apply_fn(ts.params, batch['x'], is_training = False)
    accuracy = compute_accuracy(logits, batch['y'])
    loss = cross_entropy_loss(logits, batch['y'])
    return accuracy, loss



def train(
    key             : random.PRNGKey,
    ts              : TrainState,
    ds              : Dataset,
    batch_size      : int, 
    nSteps          : int
    )-> TrainState:

    # Training loop
    for iStep in range(nSteps):
        # Training
        batch_indices = random.randint(key, batch_size, 0, len(ds))
        key, subkey = random.split(key)
        batch = ds[batch_indices]
        ts = train_step(ts, batch)   

        if iStep%100==0:
            idx = random.randint(subkey, 1000, 0, len(ds))
            accuracy, loss = eval_step(ts, ds[idx]) 
            print(f"Batches: {iStep},\tTrain Acc: {accuracy:.2%},\tloss: {loss:.4f}")    
    return ts




###################
#### ENSEMBLES ####
###################

## It would be nice to keep track of multiple seeeds at once.

def init_ensemble(
    key             : random.PRNGKey,
    model           : nn.Module,
    input_shape     : Sequence[int],
    ensemble_size   : int
    ):
    """
    Returns a PyTree of parameters with one extra leading 
    dimension at each leaf, one row per ensemble member.
    Can be applied using jax.vmap(model.apply)(params) to
    an ensemble number of batches or by 
    jax.vmap(model.apply, in_axes = (0, None)) for a single batch.
    """
    batch_x = jnp.ones([1, *input_shape])
    keys = random.split(key, ensemble_size)
    
    def init_fn(rng):
        return model.init(rng, batch_x)

    return jax.vmap(init_fn)(keys)

from functools import partial

#@partial(jax.jit, static_argnums=(2,3,4))
def get_batches(
    key             : random.PRNGKey,
    data            : dict,  # the dataset as a dict
    data_size       : int,
    batch_size      : int,
    ensemble_size   : int
):
    """
    Selects a random collection of batches of data array X.
    Can also be applied to a dictionary with jax.Array's as values. 
    Returns batches of labeled data of shape (E, B, *input_shape).
    If applied with the same key to the label array, it returns 
    the corresponding labels in the same order.
    """
    N = batch_size*ensemble_size    # number of total data samples
    idx = random.randint(key, (N,), 0, data_size)
    
    f = lambda col: col[idx].reshape(ensemble_size, batch_size, *col[0].shape)
    data_batch = jax.tree_map(f, data)

    return data_batch

def ensemble_cross_entropy_loss(Elogits, Elabels):
    """
    The ensemble loss is calculated by summing over each ensemble member.
    There is no interaction between different ensemble members, so the
    gradient with respect to the parameters minimize each model separately.
    """
    return jnp.sum(jax.vmap(cross_entropy_loss)(Elogits, Elabels))



## Training
from flax import struct

class EnsembleTrainState(TrainState):
    E               : int
    apply_single    : Callable  = struct.field(pytree_node=False)  # applying all members of the ensemble to a single batch.


def ensemble_create_train_state(
    key             : random.PRNGKey,
    model           : nn.Module, 
    input_shape     : Sequence[int],
    optimizer       : optax.GradientTransformation,
    ensemble_size   : int
) -> EnsembleTrainState:
    
    ensemble_params = init_ensemble(key, model, input_shape, ensemble_size)
    
    return EnsembleTrainState.create(
        apply_fn = jax.vmap(jax.jit(model.apply)), 
        apply_single = jax.vmap(jax.jit(model.apply), in_axes=(0,None)), 
        params = ensemble_params,
        tx = optimizer, 
        E = ensemble_size)

@jax.jit
def ensemble_train_step(
    ets          : EnsembleTrainState,   # ensemble TrainState
    X_batches   : jax.Array,    # shape (E, B, *input_shape)
    y_batches   : jax.Array     # shape (E, B, K)   
):
    """
    Given a TrainState object ts with ensemble parameters 
    (leaves have extra leading dimension of size E) 
    and distinct batches of data & label per ensemble member
    it updates the TrainState according to the optimizer ts.tx.
    """
    def loss_fn(ensemble_params):
        logits_batches = ets.apply_fn(ensemble_params, X_batches) # shape (E, B, K)
        return ensemble_cross_entropy_loss(logits_batches, y_batches)
    
    grads = jax.grad(loss_fn)(ets.params)
    ets = ets.apply_gradients(grads = grads)
    return ets

from time import perf_counter

def ensemble_train(
    key         : random.PRNGKey,
    ets         : EnsembleTrainState,
    ds          : Dataset,
    nBatch      : int,
    nSteps      : int
):
    
    gbt = 0.
    tst = 0.
    evt = 0.
    for iStep in range(nSteps):

        key, subkey = random.split(key)
        t1 = perf_counter()
        #same key gets the corresponding batches
        batches = get_batches(key, ds[:], len(ds), nBatch, ets.E)
        t2 = perf_counter()
        ets = ensemble_train_step(ets, batches['x'], batches['y'])
        t3 = perf_counter()
        if iStep %100 ==0:
            idx = random.randint(subkey, 1000, 0, len(ds))
            batch  = ds[idx]
            acc, loss = ensemble_eval(ets, batch)
            print(f"Batches: {iStep}, \t Ensemble Train acc: {acc:.2%}, \t loss: {loss:.4f}")
        t4 = perf_counter()
        evt += t4 - t3
        tst += t3 - t2
        gbt += t2 - t1

    print(f"get batches time {gbt:.2f}, \t train_step time {tst:.2} evaluation time {evt:.2}")
 
    return ets


## Evaluation

def ensemble_most_common_prediction_accuracy(
    ets                  : EnsembleTrainState, # classifier model: (B, input_shape) --> (B,K)
    X_batch             : jax.Array, # array of shape (B, *input_shape)
    y_batch             : jax.Array # integer array of shape (B,)
    ):
    logits = ets.apply_single(ets.params, X_batch)  #shape (E, B, K)
    predictions = jnp.argmax(logits, axis =-1) # shape (E, B)
    @jax.jit
    @jax.vmap
    def find_modes(array):
        unique_elts, counts = jnp.unique(array, size = ets.E, return_counts=True)
        return unique_elts[jnp.argmax(counts)]
    mcps = find_modes(predictions.T) #most common predictions per ensemble member
    return jnp.mean(mcps==y_batch)

def ensemble_accuracies(ets :EnsembleTrainState, batch:Dataset):
    logits = ets.apply_single(ets.params, batch['x'])
    predictions = jnp.argmax(logits, axis = -1)
    return jnp.mean( predictions == jnp.broadcast_to(batch['y'], (ets.E, *batch['y'].shape )), axis =-1 )


def ensemble_eval(ts : EnsembleTrainState, batch : Dataset):
 
    acc = ensemble_most_common_prediction_accuracy(ts, batch['x'], batch['y'])
    logits = ts.apply_single(ts.params, batch['x'])
    loss = ensemble_cross_entropy_loss(logits, jnp.broadcast_to(batch['y'], (ts.E, *batch['y'].shape)))
 
    return acc, loss
