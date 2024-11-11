import create_datasets, nets, langevin
import jax
import jax.numpy as jnp
import jax.random as random
import pickle
import os
import flax.serialization

key = random.PRNGKey(42)
MNIST = create_datasets.get_MNIST('train')
MNIST_test = create_datasets.get_MNIST('test')

lenet = nets.LeNet5()

# if already trained a model, then just load it 
model_path = 'params/lenet_MNIST_B128_N10000.pkl'

if os.path.exists(model_path):
    # load the parameters from file
    with open(model_path, 'rb') as f:
        params = lenet.init(key, MNIST[:2]['image'])
        params = flax.serialization.from_bytes(params, pickle.load(f))

    # create the TrainState with those parameters
    tx = nets.optax.sgd(learning_rate = 0.1, momentum = 0.9)
    ts = nets.TrainState.create(apply_fn=lenet.apply, params = params, tx = tx)

else:
    # initialize the parameters, train the net
    ts = nets.create_train_state(key, lenet, MNIST[:2]['image'], 0.1, 'sgd')
    ts = nets.train(key, ts, MNIST, 128, 10000) # each data point is seen about 2 times
    
    #load the parameters for later use
    with open(model_path, 'wb') as f:
        pickle.dump(flax.serialization.to_bytes(ts.params), f)

test_accuracy, test_loss = nets.eval_step(ts, MNIST_test[:])
print(f"On the test set: Accuracy {test_accuracy:.2%} \t loss {test_loss:.4f}")

idx = random.randint(key, 1024, 0, 60_000)
F, gradF = nets.F_function(lenet, MNIST[idx], beta = 100.) # cold temperature 0.01 is taken
hypsF = F, gradF, 1e-4

params = ts.params
params_state = key, params

(last_key, last_params), traj_params = langevin.pytree_MALA_chain(params_state, hypsF, 300)

print(jax.vmap(F)(traj_params))

traj_path = 'params/traj_params_lenet_MNIST_B1024_N300'