import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence
from datasets import Dataset
from math import prod

############################
#### Neural Net Modules ####
############################


def _flatten(x:jax.Array):
    """
    Flattens an input and adds a batch dimension if non-existent.
    (B, H, W,C) --> (B, H*W*C)
    (H, W, C)   --> (1, H*W,C)
    (B, I)      --> (B, I)
    (I,)        --> (1, I)
    The image inputs are assumed to have a color channel. 
    For example MNIST images should be input as (28,28, 1)
    """
    if x.ndim == 4:
        x = x.reshape(*x.shape[:-3], -1)
    elif x.ndim == 3:
        x = x.reshape(1, -1)
    elif x.ndim == 1:
        x = jnp.expand_dims(x, 0)
    return x


# Standard feed forward multi-layer perceptron
class MLP(nn.Module):
    features    : Sequence[int]
    
    @nn.compact
    def __call__(self, x : jax.Array):
        
        # reshape to vector (Batch dimension is kept the same)
        x = _flatten(x)

        for feature in self.features[:-1]:
            x = nn.Dense(feature)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x


class MLP_with_dropout(nn.Module):
    features        : Sequence[int]
    dropout_rate    : float | Sequence[float]
    
    @nn.compact     
    def __call__(self, x, is_training: bool = True):
        rates = [self.dropout_rate]*len(self.features) \
            if isinstance(self.dropout_rate, float)  \
            else self.dropout_rate + (self.dropout_rate[-1],)*len(self.features)

        x = _flatten(x)        
        for i, feature in enumerate(self.features[:-1]):
            x = nn.Dense(feature)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate = rates[i], deterministic=not is_training)(x)
        x = nn.Dense(self.features[-1])(x)

        return x



# Classical LeNet5 architecture
class LeNet5(nn.Module):
    num_classes: int = 10  # Number of output classes,
    conv_features   : Sequence[int] = 6, 16     # two convolutional layers
    fc_features     : Sequence[int] = 120, 84  # followed by two fully connected layers
    paddings        : Sequence[str] = 'SAME', 'VALID' 
    # original architecture assumes input of size 32 x 32 
    # this is equivalent to padding the images by 2 pixels 
    # in each direction. Thus the first layer gets 'SAME' padding. 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        for feature, padding in zip(self.conv_features, self.paddings):
            x = nn.Conv(features=feature, kernel_size=(5, 5), strides=(1, 1), padding = padding)(x)
            x = nn.sigmoid(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape(x.shape[0], -1)  
        
        for feature in self.fc_features:
            x = nn.Dense(features=feature)(x)
            x = nn.sigmoid(x)
        
        x = nn.Dense(features=self.num_classes)(x)
        return x
    


class ConvBlock(nn.Module):
    features        : int 
    dropout_rate    : float

    @nn.compact
    def __call__(self, x : jax.Array, is_training : bool = True):
        x = nn.Conv(self.features, kernel_size = (3,3), padding = 'SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size = (3,3), padding = 'SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))
        x = nn.Dropout(rate = self.dropout_rate, deterministic = not is_training)(x)
        return x


class CNN(nn.Module):
    cnn_features : Sequence[int]
    mlp_features : Sequence[int]
    dropout_rate : float

    @nn.compact
    def __call__(self, x, is_training:bool = True):
        
        for nFeatures in self.cnn_features:
            x = ConvBlock(features=nFeatures,  dropout_rate=self.dropout_rate)(x, is_training)
    
        x = MLP_with_dropout(features = self.mlp_features, dropout_rate = self.dropout_rate)(x, is_training)
        return x    
    




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
    
    xs = ds['x']
    ys = ds['y']

    @jax.jit
    def F(params):
        logits = model.apply(params, xs, is_training = False)
        logprobs = jax.nn.log_softmax(logits)
        @jax.vmap # for each batch member gets the label column of the B x K logits matrix.
        def index(logits, label):
            return logits[label]
        loss = -jnp.mean(index(logprobs,ys))
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
        logits = jax.vmap(partial(model.apply, x = x, is_training = False))(params_traj)
        losses = (-jax.nn.log_softmax(logits)[:, label])
        loss = jnp.mean(losses)
        return beta*(loss + const1*l1_reg(x) + const2*total_variation(x.squeeze()))
    
    return G, jax.grad(G)