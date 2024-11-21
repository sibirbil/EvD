import jax
import jax.numpy as jnp
import jax.random as random
import pickle
import flax

import vae, nets, create_datasets

MNIST = create_datasets.get_MNIST('train')

#add extra dimension and get only the images
MNISTimgs = jnp.expand_dims(MNIST['image'], axis = -1) #shape (60000, 28, 28, 1)

key = random.PRNGKey(1291)


#############
##   VAE   ##
#############
latent_dim = 10
input_shape = (28,28,1)
vae_model = vae.VAE(latent_dim=latent_dim, input_shape=input_shape)
vae_model_path = 'params/vae_MNIST_B128_N10000eta0.001momentum0.9.pkl'

with open(vae_model_path, 'rb') as f:
    vae_params = vae_model.init(key, key, jnp.ones([1, *vae_model.input_shape]))
    vae_params = flax.serialization.from_bytes(vae_params, pickle.load(f))

encoder_module = vae.VAEncoder(latent_dim)
encoder_params = {'params':vae_params['params']['encoder']}

def encoder(key : random.PRNGKey, x: jax.Array):
    mu, logvar = encoder_module.apply(encoder_params, x[jnp.newaxis,:])
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, std.shape)
    return mu + eps * std

decoder_module = vae.VADecoder(latent_dim, input_shape)
decoder_params = {'params':vae_params['params']['decoder']}
    
def decoder(z):
    return decoder_module.apply(decoder_params, z[jnp.newaxis,:])


###################
##  CLASSIFIERS  ##
###################

lenet = nets.LeNet5()
features = [512,128,32,10]
mlp = nets.MLP(features)

# CNN classifier
lenet = nets.LeNet5()
lenet_param_path = "params/lenet_MNIST_B256_N10000.pkl"
with open(lenet_param_path, 'rb') as f:
    lenet_params = lenet.init(key, MNIST[:2]['image'])
    lenet_params = flax.serialization.from_bytes(lenet_params, pickle.load(f))

# MLP classifier
features  = [512, 128,32,10]
mlp = nets.MLP(features)
mlp_param_path = "params/mlp_512-128-32-10_MNIST_B256_N5000.pkl"
with open(mlp_param_path, 'rb') as f:
    mlp_params = mlp.init(key, MNISTimgs[:2])
    mlp_params = flax.serialization.from_bytes(mlp_params, pickle.load(f))


label = 7
beta = 1.
import langevin


def G(z):
    x = jax.nn.sigmoid(decoder(z))
    mlp_logits = mlp.apply(mlp_params, x)
    lenet_logits = lenet.apply(lenet_params, x)
    loss = - (jax.nn.log_softmax(mlp_logits) + jax.nn.log_softmax(lenet_logits))[0, label]
    return beta * loss

gradG = jax.grad(G)

hypsG= G, gradG, 5e-4
z0 = encoder(key, MNISTimgs[1000])
state_z = key, z0



(last_key, last_z), traj_z = langevin.MALA_chain(state_z, hypsG, 5000)

x0 = decoder(z0)[jnp.newaxis,:]
last_x = decoder(last_z)[jnp.newaxis,:]
traj_x = jax.vmap(decoder)(traj_z)
traj_x = jnp.vstack([jnp.repeat(x0, 100, axis = 0), traj_x, jnp.repeat(last_x, 100, axis = 0)])
traj_x = jax.nn.sigmoid(traj_x)

## Tried to do it directly with pixels but it didn't work 
## so it is commented out

# def PixelG(x):
#     tot_var = nets.total_variation(x.squeeze())
#     laplacian = nets.laplacian(x.squeeze())
#     l1 = nets.l1_reg(x)
#     x =  jnp.expand_dims(x, axis = 0)
#     mlp_logits = mlp.apply(mlp_params, x)
#     lenet_logits = lenet.apply(lenet_params, x)
#     loss = - (jax.nn.log_softmax(mlp_logits) + jax.nn.log_softmax(lenet_logits))[0, label]
#     return beta*(loss + abs(l1-100) + tot_var + laplacian)


# gradPixelG = jax.grad(PixelG)
# hypsPixelG = PixelG, gradPixelG, 1e-4
# x0 =  MNISTimgs[1000]
# state_x = key, x0

# (last_key, last_x), traj_x = langevin.MALA_chain(state_x, hypsPixelG, 20000)

# traj_x = jnp.vstack([jnp.repeat(x0[jnp.newaxis,:], 100, axis = 0), traj_x, jnp.repeat(last_x[jnp.newaxis,:], 100, axis = 0)])


import plotting


# plotting.animate(traj_x.squeeze())
plotting.animate(traj_x.squeeze(), 'images/joint0to7.gif')

