import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from jax import random
from typing import Tuple
from datasets import Dataset



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


class LeNet5(nn.Module):
    num_classes: int = 10  # Number of output classes,
    conv_features   : Tuple[int] = 6,16
    fc_features     : Tuple[int] = 120, 84
    paddings        : Tuple[str] = 'SAME', 'VALID'

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


def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(labels[jnp.arange(len(predictions)),predictions])

class TrainState(train_state.TrainState):
    pass



@jax.jit
def train_step(
        state   : TrainState, 
        batch   : Tuple[jax.Array, jax.Array]
    )-> Tuple[TrainState, jnp.float_]:
    """
        performs 
    """
    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(
    state       : TrainState,
    batch
    ):
    logits = state.apply_fn(state.params, batch['image'])
    accuracy = compute_accuracy(logits, batch['label'])
    loss = cross_entropy_loss(logits, batch['label'])
    return accuracy, loss


def create_train_state(
        key             : jax.Array, 
        model           : nn.Module, 
        fake_batch      : jax.Array, 
        learning_rate   : jnp.float_,
        optimizer       : str   = 'adam'
        ):
    if optimizer == 'adam':
        tx = optax.adam(learning_rate)
    elif optimizer == 'sgd':
        tx = optax.sgd(learning_rate, momentum = 0.9)

    params = model.init(key, fake_batch)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

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
            print(f"Batches processed:{iStep}, Accuracy: {accuracy}, loss: {loss}")    
    return ts

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def show_filters(F): 
    A = jnp.max(jnp.abs(F))
    cmap = mcolors.TwoSlopeNorm(vmin=-A, vcenter=0, vmax=A)
    plt.imshow(F.reshape(28,28), cmap = 'bwr', norm = cmap)
    plt.colorbar(label = 'Value')
    plt.show()


from logistic import l1_reg


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
    params_traj,           # a pytree of parameters with an extra leading dimension
    model       : nn.Module,
    label       : int,
    beta        : jnp.float_,
    reg_const   : jnp.float_
    ):
    
    def G(x):
        lbl = jax.nn.one_hot(jnp.array(label), 10)
        logits = jax.vmap(partial(model.apply, x = jax.nn.relu(x)[jnp.newaxis, :]))(params_traj)
        losses = (-jax.nn.log_softmax(logits) @ lbl)
        loss = beta*jnp.mean(losses)
        return loss + reg_const*l1_reg(x)
    
    return G, jax.grad(G)