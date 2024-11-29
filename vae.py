import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple
from datasets import Dataset
from nets import MLP

###############################
# NON-VARIATIONAL AutoEncoder #
###############################

# architecture taken from
# https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.ipynb#scrollTo=QFr6cv6lafOt
class AEncoder(nn.Module):
    c_hid : int
    latent_dim : int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 32x32 => 16x16
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 16x16 => 8x8
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 8x8 => 4x4
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.latent_dim)(x)
        return x
    


class ADecoder(nn.Module):
    c_out : int
    c_hid : int
    latent_dim : int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*16*self.c_hid)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(features=2*self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_out, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.tanh(x)
        return x

class AutoEncoder(nn.Module):
    c_hid: int
    latent_dim : int
        
    def setup(self):
        # Alternative to @nn.compact -> explicitly define modules
        # Better for later when we want to access the encoder and decoder explicitly
        self.encoder = AEncoder(c_hid=self.c_hid, latent_dim=self.latent_dim)
        self.decoder = ADecoder(c_hid=self.c_hid, latent_dim=self.latent_dim, c_out=3)
        
    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    

    
#############
# VAE Model #
#############

class VAEncoder(nn.Module):
    latent_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))
        self.fc_mu = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)

    def __call__(self, x: jax.Array):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten before fully connected
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VADecoder(nn.Module):
    latent_dim: int
    original_shape: Tuple[int]

    def setup(self):
        self.fc = nn.Dense(7 * 7 * 64)  # Reshape target size for upsampling
        self.deconv1 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2))
        self.deconv2 = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))
        self.final_layer = nn.Conv(features= 1, kernel_size=(3, 3), padding="SAME") #change feature to 3 for color images

    def __call__(self, z):
        x = nn.relu(self.fc(z))
        x = x.reshape((x.shape[0], 7, 7, 64))
        x = nn.relu(self.deconv1(x))
        x = nn.relu(self.deconv2(x))
        x = self.final_layer(x)
        return x  # Output in range [0, 1]

class VAE(nn.Module):
    latent_dim: int
    input_shape: Tuple[int]

    def setup(self):
        self.encoder = VAEncoder(self.latent_dim)
        self.decoder = VADecoder(self.latent_dim, self.input_shape)

    def reparametrize(self, key, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        return mu + eps * std  # latent variable z

    def __call__(self, key, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(key, mu, logvar)
        x_recon_logits = self.decoder(z)
        return x_recon_logits, mu, logvar



########################
##  TABULAR DATA VAE  ##
########################

from nets import MLP
from math import log2

class TabularVAE(nn.Module):
    latent_dim      : int
    input_dim       : int

    def setup(self):
        num_hl = int(log2(self.input_dim/self.latent_dim)) + 2 #number of hidden layers

        # start at 2 to 4 times the input dimension, half at each layer,
        # we repeat 2*latent_dim twice since the last one corresponds to two heads 
        # in the latent space, one for the mean, other for log-variance.
        features = [self.latent_dim*2**n for n in range(num_hl, 0, -1)] + [2*self.latent_dim]
        
        self.encoder = MLP(features)        # the mu and logvar are stacked, need to be reshaped
        self.decoder = MLP(features[-2::-1] + [self.input_dim])

    def reparametrize(self, key, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        return mu + eps * std
    
    def __call__(self, key, x):
        mu, logvar = self.encoder(x).reshape(2,-1)  # two heads were stacked
        z = self.reparametrize(key, mu, logvar)
        x_recon_logits = self.decoder(z)
        return x_recon_logits, mu, logvar

############
# TRAINING #
############

class TrainState(train_state.TrainState):
    latent_dim : int
    input_shape : Tuple[int]


# Training state
def create_train_state(
    key             : random.PRNGKey, 
    model           : VAE, 
    tx              : optax.GradientTransformation # optimizer
    ):
    key, subkey = jax.random.split(key)
    params = model.init(key, subkey, jnp.ones([1, *model.input_shape]))
    ts = TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=tx, 
        latent_dim = model.latent_dim,
        input_shape = model.input_shape)
    return ts



# Loss functions
def BCE(
    logits      : jax.Array, #shape (B, H, W, C)
    batch       : jax.Array #shape (B, H, W, C)
    ):
    cross_ents = jnp.maximum(0., logits) + jnp.logaddexp(0, -jnp.abs(logits)) - logits * batch 
    #summ in all pixels and channels, average over the batch dimension
    bce = jnp.mean(jnp.sum(cross_ents, axis = [1,2,3]))
    return bce 

def KL(mu,logvar):
    kl_divs = 0.5 * (-1. - logvar + mu**2 + jnp.exp(logvar))
    kl_loss = jnp.mean(jnp.sum(kl_divs, axis=1))  
    return kl_loss

def compute_vae_loss(
    key     : random.PRNGKey,
    ts      : TrainState, 
    batch   : Dataset
    ):
    recon_logits, mu, logvar = ts.apply_fn(ts.params, key, batch)
    recon_loss = BCE(recon_logits, batch)
    kl_loss = KL(mu, logvar)
    return recon_loss, kl_loss

@jax.jit
def train_step(
    key         : random.PRNGKey,
    ts          : TrainState, 
    batch       : Dataset
    ):

    def loss_fn(params):
        recon_logits, mu, logvar = ts.apply_fn(params, key, batch)
        recon_loss = BCE(recon_logits, batch)
        kl_loss = KL(mu, logvar)
        return recon_loss + kl_loss
    
    grads = jax.grad(loss_fn)(ts.params)
    ts = ts.apply_gradients(grads=grads)
    return ts



def train(
    key     : random.PRNGKey,                
    ts      : TrainState, # Train State
    ds      : Dataset,
    nBatch  : int, 
    nSteps  : int
    ):
        # Training loop
    for iStep in range(nSteps):
        # Training
        key, subkey = random.split(key)
        batch_indices = random.randint(key, nBatch, 0, len(ds))
        batch = ds[batch_indices]
        ts = train_step(subkey, ts, batch)   

        if iStep%100==0:
            idx = random.randint(subkey, 1000, 0, len(ds))
            recon_loss, kl_loss = compute_vae_loss(key, ts, ds[idx]) 
            print(f"Batches: {iStep},\tReconstruction loss {recon_loss:.4f},\tKL loss: {kl_loss:.4f}")    
    return ts



def get_decoder(ts: TrainState):
    decoder = VADecoder(ts.latent_dim, ts.input_shape)
    decoder_params = {'params':ts.params['params']['decoder']}
    
    def decoder_fn(z):
        return decoder.apply(decoder_params, z[jnp.newaxis,:])

    return decoder_fn

def get_encoder(ts: TrainState):
    encoder = VAEncoder(ts.latent_dim)
    encoder_params = {'params':ts.params['params']['encoder']}

    def encoder_fn(key : random.PRNGKey, x: jax.Array):
        mu, logvar = encoder.apply(encoder_params, x[jnp.newaxis,:])
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        return mu + eps * std
    
    return encoder_fn


def G_function(
    ts                  : TrainState,
    model               : nn.Module,        #classifier model
    params,                                 #params of classifier model
    label               : int,
    beta                : jnp.float_
):

    decoder_fn = get_decoder(ts)
    
    def G(z):
        lbl = jax.nn.one_hot(jnp.array(label), 10)
        x = decoder_fn(z)
        logits = model.apply(params, jax.nn.sigmoid(x))
        loss = - jax.nn.log_softmax(logits) @ lbl
        return beta*loss[0]
    
    return G, jax.grad(G)