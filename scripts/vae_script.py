import jax
import jax.numpy as jnp
import jax.random as random
import create_datasets

key = random.PRNGKey(234)
MNIST = create_datasets.get_MNIST('train')

#add extra dimension and get only the images
MNISTimgs = jnp.expand_dims(MNIST['image'], axis = -1) #shape (60000, 28, 28, 1)

import vae

ts = vae.ts

ts = vae.train(key, ts, MNISTimgs, 128, 5000)

def reconstruction(x):
    if len(x.shape) ==3:
        x = x[jnp.newaxis, :]
    logits, mu, logvar = vae.model.apply(ts.params, key, x[jnp.newaxis, :])
    return jax.nn.sigmoid(logits[0,:,:,0])

import plotting

plotting.show_mnist_example(MNISTimgs[1242])
plotting.show_mnist_example(reconstruction(MNISTimgs[1242]))