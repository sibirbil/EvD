import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from jax import random
from typing import Tuple, Callable
from datasets import Dataset

from math import prod

############################
#### Neural Net Modules ####
############################

# Standard feed forward multi-layer perceptron
class MLP(nn.Module):
    features    : Tuple[int]
    input_shape : int | Tuple[int] = (28,28)

    @nn.compact
    def __call__(self, x : jax.Array):
        
        # reshape to vector (Batch dimension is kept the same)
        x = x.reshape(-1, prod(self.input_shape))

        for feature in self.features[:-1]:
            x = nn.Dense(feature)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

# Classical LeNet5 architecture
class LeNet5(nn.Module):
    num_classes: int = 10  # Number of output classes,
    conv_features   : Tuple[int] = 6, 16     # two convolutional layers
    fc_features     : Tuple[int] = 120, 84  # followed by two fully connected layers
    paddings        : Tuple[str] = 'SAME', 'VALID' 
    # original architecture assumes input of size 32 x 32 
    # this is equivalent to padding the images by 2 pixels 
    # in each direction. Thus the first layer gets 'SAME' padding. 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        for feature, padding in zip(self.conv_features, self.paddings):
            x = nn.Conv(features=feature, kernel_size=(5, 5), strides=(1, 1), padding = padding)(x)
            x = nn.sigmoid(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))  
        
        for feature in self.fc_features:
            x = nn.Dense(features=feature)(x)
            x = nn.sigmoid(x)
        
        x = nn.Dense(features=self.num_classes)(x)
        return x

###################
##### LOSSES ######
###################

## this was for one hot encoded labels (less memory efficient)
# def cross_entropy_loss(logits, labels):
#     return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

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
    pass


def create_train_state(
        key             : jax.Array, 
        model           : nn.Module, 
        fake_batch      : jax.Array, 
        learning_rate   : optax.ScalarOrSchedule,
        optimizer       : str   = 'adam'
        ):
    if optimizer == 'adam':
        tx = optax.adam(learning_rate)
    elif optimizer == 'sgd':
        tx = optax.sgd(learning_rate, momentum = 0.9)

    params = model.init(key, fake_batch)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(
        state   : TrainState, 
        batch   : dict,
    )-> TrainState:
    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(ts : TrainState, batch : Dataset):

    logits = ts.apply_fn(ts.params, batch['image'])
    accuracy = compute_accuracy(logits, batch['label'])
    loss = cross_entropy_loss(logits, batch['label'])
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

## It would be nice to keep track of 

def init_ensemble(
    key             : random.PRNGKey,
    model           : nn.Module,
    input_size      : Tuple[int],
    ensemble_size   : int
    ):
    """
    Returns a PyTree of parameters with one extra leading 
    dimension at each leaf, one row per ensemble member.
    Can be applied using jax.vmap(model.apply)(params) to
    an ensemble number of batches or by 
    jax.vmap(model.apply, in_axes = (0, None)) for a single batch.
    """
    batch_x = jnp.ones([1, *input_size])
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
    input_size      : Tuple[int],
    optimizer       : optax.GradientTransformation,
    ensemble_size   : int
) -> EnsembleTrainState:
    
    ensemble_params = init_ensemble(key, model, input_size, ensemble_size)
    
    return EnsembleTrainState.create(
        apply_fn = jax.vmap(model.apply), 
        apply_single = jax.vmap(model.apply, in_axes=(0,None)), 
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


def ensemble_train(
    key         : random.PRNGKey,
    ets         : EnsembleTrainState,
    ds          : Dataset,
    nBatch      : int,
    nSteps      : int
):
    
    for iStep in range(nSteps):
        key, subkey = random.split(key)

        #same key gets the corresponding batches
        batches = get_batches(key, ds[:], len(ds), nBatch, ets.E)
        
        ets = ensemble_train_step(ets, batches['image'], batches['label'])

        if iStep %100 ==0:
            idx = random.randint(subkey, 1000, 0, len(ds))
            batch  = ds[idx]
            acc, loss = ensemble_eval(ets, batch)
            print(f"Batches: {iStep}, \t Ensemble Train acc: {acc:.2%}, \t loss: {loss:.4f}")

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


def ensemble_eval(ts : EnsembleTrainState, batch : Dataset):
 
    acc = ensemble_most_common_prediction_accuracy(ts, batch['image'], batch['label'])
    logits = ts.apply_single(ts.params, batch['image'])
    loss = ensemble_cross_entropy_loss(logits, jnp.broadcast_to(batch['label'], (ts.E, *batch['label'].shape)))
 
    return acc, loss



#########################################
##  SOME REGULARIZERS ON PIXEL VALUES  ##
#########################################

from logistic import l1_reg

def total_variation(image):
    """
    Total variation measure 
    """
    dx, dy = jnp.gradient(image)
    return jnp.sum(jnp.abs(dx) + jnp.abs(dy))

def laplacian(image):
    dx, dy = jnp.gradient(image)
    ddx, _ = jnp.gradient(dx)
    _, ddy = jnp.gradient(dy)
    return jnp.sum(jnp.abs(ddx + ddy))


# the F and the G functions to be used in MALA or VI.
# for finding distributions of parameters or data. 

def F_function(
    model   : nn.Module,
    ds      : Dataset,
    beta    : jnp.float_ #inverse temperature 
):
    xs = ds['image']
    ys = ds['label']

    @jax.jit
    def F(params):
        logits = model.apply(params, xs)
        a = - jax.nn.log_softmax(logits)
        loss = jnp.mean(jnp.sum(a*ys, axis = 1))
        return beta*loss
    
    return F, jax.grad(F)
        

from functools import partial

def G_function(
    params_traj,            # a pytree of parameters with an extra leading dimension
    model     : nn.Module,
    label     : int,        # target label (an integer form 0 to 9)
    beta      : jnp.float_, # inverse temperature multiplying all of 
    const1    : jnp.float_, # constant in front of l1 regularization
    const2    : jnp.float_  # constant multiplying the total variation regularization
    ):
    
    @jax.jit
    def G(x):
        lbl = jax.nn.one_hot(jnp.array(label), 10)
        logits = jax.vmap(partial(model.apply, x = x[jnp.newaxis, :]))(params_traj)
        losses = (-jax.nn.log_softmax(logits) @ lbl)
        loss = jnp.mean(losses)
        return beta*(loss + const1*l1_reg(x) + const2*total_variation(x))
    
    return G, jax.grad(G)