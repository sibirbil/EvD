import create_datasets, nets, langevin, train
import jax
import jax.numpy as jnp
import jax.random as random
import pickle
import os
import flax.serialization
import optax
from flax import linen as nn
from typing import Sequence


TrainBatchSize = 128
TrainSteps = 10_000
F_BatchSize = 32
F_Steps = 16 
F_Temp = 100.
F_eta = 1e-4        #step size in hypsF
G_Steps = 1000
G_Temp = 1000.
G_l1reg = .00025           # ell_1 regularizer constant
G_tvreg = .001           # total_variation regularization constant
G_eta = 0.01/G_Temp       #step size in hypsG
G_label = 5

key = random.PRNGKey(41)
dropout_key, init_key, train_key, z0_key, encoder_key, MALA_key = random.split(key, 6)
MNIST = create_datasets.get_MNIST('train')
MNIST_test = create_datasets.get_MNIST('test')

features = [1024,128,10] 
dropout_rate = 0.2
mlp = nets.MLP_with_dropout(features, dropout_rate)

mlp_model_path = 'params/MNIST_mlp_' + ('-'.join(map(str, mlp.features))) + '_do' + str(mlp.dropout_rate) + '_N' + str(TrainSteps) + '.pkl'
exp_decay_schedule = optax.schedules.exponential_decay(0.01,100, 0.9,500, True, end_value = 1e-5)
adam = optax.adam(learning_rate = exp_decay_schedule)

if os.path.exists(mlp_model_path):
    # load the parameters from file
    with open(mlp_model_path, 'rb') as f:
        params = mlp.init(key, MNIST[:2]['x'])
        params = flax.serialization.from_bytes(params, pickle.load(f))

    # create the TrainState with those parameters
    mts = train.TrainState.create(apply_fn=mlp.apply, params = params, tx = adam, rng_key = dropout_key)

else:
    # initialize the parameters, train the net
    mts = train.create_train_state(key, mlp, (28, 28,1), adam)
    mts = train.train(key, mts, MNIST, 128, TrainSteps)

    #load the parameters for later use
    with open(mlp_model_path, 'wb') as f:
        pickle.dump(flax.serialization.to_bytes(mts.params), f)


test_accuracy, test_loss = train.eval_step(mts, MNIST_test[:])
print(f"MLP On the test set: Accuracy {test_accuracy:.2%} \t loss {test_loss:.4f}")


cnn = nets.CNN(cnn_features=[32,64], mlp_features = [128,10], dropout_rate=0.2)
cts = train.create_train_state(init_key, cnn, MNIST[0]['x'].shape, adam)


cnn_model_path = 'params/MNIST_cnn_' + ('-'.join(map(str, cnn.cnn_features))) + '_mlp_' +('-'.join(map(str, cnn.mlp_features))) \
    + '_do' + str(cnn.dropout_rate) + '_N' + str(TrainSteps) + '.pkl'


if os.path.exists(cnn_model_path):
    # load the parameters from file
    with open(cnn_model_path, 'rb') as f:
        params = cnn.init(key, MNIST[:2]['x'])
        params = flax.serialization.from_bytes(params, pickle.load(f))

    # create the TrainState with those parameters
    cts = train.TrainState.create(apply_fn= cnn.apply, params = params, tx = adam, rng_key = dropout_key)

else:
    # initialize the parameters, train the net
    cts = train.create_train_state(key, cnn, (28, 28,1), adam)
    cts = train.train(key, cts, MNIST, TrainBatchSize, TrainSteps)

    #load the parameters for later use
    with open(cnn_model_path, 'wb') as f:
        pickle.dump(flax.serialization.to_bytes(cts.params), f)



test_accuracy, test_loss = train.eval_step(cts, MNIST_test[:])
print(f"CNN On the test set: Accuracy {test_accuracy:.2%} \t loss {test_loss:.4f}")


from scripts.vae_script import encoder as vae_encoder
from scripts.vae_script import decoder as vae_decoder

def G_function(
    model     : nn.Module,
    params,
    target    : int,
    beta      : jnp.float_              # inverse temperature
    ):
    
    def G(x):
        logits = model.apply(params, x, is_training = False)
        loss = -jax.nn.log_softmax(logits.squeeze())[target]
        return beta*loss

    return jax.jit(G)        

target1 = 5
target2 = 2
betaG = 10.
etaG  = 0.001/betaG

G1 = G_function(mlp, mts.params, target1, betaG)
G2 = G_function(cnn, cts.params, target2, betaG)

def G(z):
    x = vae_decoder(z)
    g1 = G1(x)
    g2 = G2(x)
    return g1 + g2 
hypsG = G, jax.grad(G), etaG

def G_function_counterfactual(
    model       : nn.Module,
    params,
    anchor      : jax.Array,
    target      : int,
    beta        : jnp.float_
    ):

    G = G_function(model, params, target, beta)

    def main(z):
        x = vae_decoder(z)
        return G(x)

    def localizer(z):
        data = vae_decoder(z)
        l2_reg = jnp.mean(jnp.square(data - anchor))
        return l2_reg

    return jax.jit(lambda z : localizer(z) + main(z))

import plotting


#z0 = vae_encoder(encoder_key, MNIST[125]['x'])
z0 = random.normal(z0_key, shape = (10,))
state_z = MALA_key, z0

_, traj_z = langevin.MALA_chain(state_z, hypsG, 10000)

traj_x = jax.nn.sigmoid(jax.vmap(vae_decoder)(traj_z))



#save_location = 'images/contrast_mlp_'+ str(target1) + '_cnn_' + str(target2) + '.png'
#plotting.animate(traj_x)


# traj_path = 'params/traj_params_mlp_' + '-'.join(map(str, features)) + '_MNIST_B' +str(F_BatchSize) + '_N' + \
#             str(F_Steps)+ '_T' + str(F_Temp) +'_eta' + str(F_eta) + '.pkl'

# # create F, and hypsF
# idx = random.randint(key, F_BatchSize, 0, 60_000)
# F, gradF = nets.F_function(model, MNIST[idx], beta = F_Temp) 
# hypsF = F, gradF, 1e-4


# if os.path.exists(traj_path):
#     # load the parameters from file
#     with open(traj_path, 'rb') as f:
#         placeholder_tree = model.init(key, MNIST[:2]['x'])
#         params_traj = flax.serialization.from_bytes(placeholder_tree, pickle.load(f))

# else:
#     #compute parameters load the parameters for later use
#     params_state = key, ts.params
#     _, params_traj = langevin.MALA_chain(params_state, hypsF, F_Steps)
#     with open(traj_path, 'wb') as f:
#         pickle.dump(flax.serialization.to_bytes(params_traj), f)


# print(jax.vmap(F)(params_traj))


