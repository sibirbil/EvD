import os
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import flax
import pickle

import create_datasets

key = random.PRNGKey(234)
MNIST = create_datasets.get_MNIST('train')

#add extra dimension and get only the images
MNISTimgs = MNIST['x'] #shape (60000, 28, 28, 1)

import vae


TrainBatchSize = 128
TrainSteps = 10_000
momentum = 0.9
eta = 1e-3


latent_dim = 10
input_shape = (28, 28, 1)  # Adjust to (32, 32, 3) for CIFAR-10
vae_model = vae.VAE(latent_dim=latent_dim, input_shape=input_shape)
key = jax.random.PRNGKey(42)
key, subkey = random.split(key)
lr_scheduler  = eta
#lr_scheduler = optax.constant_schedule(eta)
tx = optax.sgd(learning_rate = lr_scheduler, momentum = momentum)

# if already trained a model, then just load it 
vae_model_path = 'params/vae_MNIST_B'+str(TrainBatchSize) + '_N' + str(TrainSteps) + 'eta' + str(eta) + 'momentum' + str(momentum) + '.pkl'

if os.path.exists(vae_model_path):
    # load the parameters from file
    with open(vae_model_path, 'rb') as f:
        params = vae_model.init(key, subkey, jnp.ones([1, *vae_model.input_shape]))
        params = flax.serialization.from_bytes(params, pickle.load(f))

    # create the TrainState with those parameters
    ts = vae.TrainState.create(
        apply_fn=vae_model.apply, 
        params = params, 
        tx = tx, 
        latent_dim = vae_model.latent_dim, 
        input_shape = vae_model.input_shape
        )

else:
    # initialize the parameters, train the net
    ts = vae.create_train_state(key, vae_model, tx)
    ts = vae.train(key, ts, MNISTimgs, TrainBatchSize, TrainSteps)

    #load the parameters for later use
    with open(vae_model_path, 'wb') as f:
        pickle.dump(flax.serialization.to_bytes(ts.params), f)


def reconstruction(x :jax.Array, ts: vae.TrainState):
    if len(x.shape) ==3:
        x = x[jnp.newaxis, :]
    logits, mu, logvar = ts.apply_fn(ts.params, key, x[jnp.newaxis, :])
    return jax.nn.sigmoid(logits[0,:,:,0])

# import plotting

# plotting.image_show(MNISTimgs[1242])
# plotting.image_show(reconstruction(MNISTimgs[1242], ts))


# # CNN classifier
# lenet = nets.LeNet5()
# lenet_param_path = "params/lenet_MNIST_B256_N10000.pkl"
# with open(lenet_param_path, 'rb') as f:
#     lenet_params = lenet.init(key, MNIST[:2]['x'])
#     lenet_params = flax.serialization.from_bytes(lenet_params, pickle.load(f))

# # MLP classifier
# features  = [512, 128,32,10]
# mlp = nets.MLP(features)
# mlp_param_path = "params/mlp_512-128-32-10_MNIST_B256_N5000.pkl"
# with open(mlp_param_path, 'rb') as f:
#     mlp_params = mlp.init(key, MNISTimgs[:2])
#     mlp_params = flax.serialization.from_bytes(mlp_params, pickle.load(f))

# G, gradG = vae.G_function(ts, lenet, lenet_params, 5, 1.)

# import langevin

encoder = vae.get_encoder(ts)
decoder = vae.get_decoder(ts)

# x = MNISTimgs[1242]
# z = encoder(key, x)
# state_z = key, z
# hypsG = G, gradG, 0.001


# (last_key, last_z), traj_z = langevin.MALA_chain(state_z, hypsG, 30000)


# extended_traj_z = jnp.vstack([jnp.repeat(z[jnp.newaxis,:], 100, axis =0), traj_z, jnp.repeat(jnp.expand_dims(last_z, axis =1), 100, axis = 0)])

# frames = jax.nn.sigmoid(jax.vmap(decoder)(extended_traj_z)).squeeze()

# plotting.animate(frames, save_filename="images/lenet1to5.gif")