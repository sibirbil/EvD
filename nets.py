import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from jax import random
from typing import Tuple
from datasets import Dataset


############################
#### Neural Net Modules ####
############################

# Standard feed forward multi-layer perceptron
class MLP(nn.Module):
    features: list

    @nn.compact
    def __call__(self, x : jax.Array):
        
        x = x.reshape(x.shape[0],-1)

        for feature in self.features[:-1]:
            x = nn.Dense(feature)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

# Classical LeNet5 architecture
class LeNet5(nn.Module):
    num_classes: int = 10  # Number of output classes,
    conv_features   : Tuple[int] = 6,16     # two convolutional layers
    fc_features     : Tuple[int] = 120, 84  # followed by two fully connected layers
    paddings        : Tuple[str] = 'SAME', 'VALID' 
    # original architecture assumes input of size 32 x 32 
    # this is equivalent to padding the images by 2 pixels 
    # in each direction. Thus the first layer gets 'SAME' padding. 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # adding an explicit channel dimension if not there
        # batch dimension is always assumed to be there
        if len(x.shape) == 3:
            x = x[..., jnp.newaxis] 
        
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

def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(labels[jnp.arange(len(predictions)),predictions])


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
        batch   : Tuple[jax.Array, jax.Array]
    )-> Tuple[TrainState, jnp.float_]:
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


def F_function(
    model   : nn.Module,
    ds      : Dataset,
    beta    : jnp.float_ #inverse temperature 
):
    xs = ds['image']
    ys = ds['label']

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