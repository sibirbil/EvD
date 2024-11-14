import create_datasets, nets, langevin
import jax
import jax.numpy as jnp
import jax.random as random
import pickle
import os
import flax.serialization
from time import perf_counter

TrainBatchSize = 256
TrainSteps = 10_000
F_BatchSize = 32
F_Steps = 32 
F_Temp = 100.
F_eta = 1e-4        #step size in hypsF
G_Steps = 5000
G_Temp = 10000.
G_l1reg = .01           # ell_1 regularizer constant
G_tvreg = .02           # total_variation regularization constant
G_eta = 1e-4        #step size in hypsG
G_label = 1

key = random.PRNGKey(31)
MNIST = create_datasets.get_MNIST('train')
MNIST_test = create_datasets.get_MNIST('test')

model = nets.LeNet5()

# if already trained a model, then just load it 
model_path = 'params/lenet_MNIST_B'+str(TrainBatchSize) + '_N' + str(TrainSteps) + '.pkl'

start = perf_counter()
if os.path.exists(model_path):
    # load the parameters from file
    with open(model_path, 'rb') as f:
        params = model.init(key, MNIST[:2]['image'])
        params = flax.serialization.from_bytes(params, pickle.load(f))

    # create the TrainState with those parameters
    tx = nets.optax.sgd(learning_rate = 0.1, momentum = 0.9)
    ts = nets.TrainState.create(apply_fn=model.apply, params = params, tx = tx)

else:
    # initialize the parameters, train the net
    ts = nets.create_train_state(key, model, MNIST[:2]['image'], 0.1, 'sgd')
    ts = nets.train(key, ts, MNIST, TrainBatchSize, TrainSteps) 

    #load the parameters for later use
    with open(model_path, 'wb') as f:
        pickle.dump(flax.serialization.to_bytes(ts.params), f)
end = perf_counter()
print(f"time to load the parameters {end -start}")


test_accuracy, test_loss = nets.eval_step(ts, MNIST_test[:])
print(f"On the test set: Accuracy {test_accuracy:.2%} \t loss {test_loss:.4f}")


traj_path = 'params/traj_params_lenet_MNIST_B' +str(F_BatchSize) + '_N' + \
            str(F_Steps)+ '_T' + str(F_Temp) +'_eta' + str(F_eta) + '.pkl'

# create F, and hypsF
idx = random.randint(key, F_BatchSize, 0, 60_000)
F, gradF = nets.F_function(model, MNIST[idx], beta = F_Temp) 
hypsF = F, gradF, 1e-4


start = perf_counter()
if os.path.exists(traj_path):
    # load the parameters from file
    with open(traj_path, 'rb') as f:
        placeholder_tree = model.init(key, MNIST[:2]['image'])
        params_traj = flax.serialization.from_bytes(placeholder_tree, pickle.load(f))

else:
    #compute parameters load the parameters for later use
    params_state = key, ts.params
    _, params_traj = langevin.MALA_chain(params_state, hypsF, F_Steps)
    with open(traj_path, 'wb') as f:
        pickle.dump(flax.serialization.to_bytes(params_traj), f)
end = perf_counter()
print(f"time to load F langevin parameter trajectory {end -start}")




start = perf_counter()
print(jax.vmap(F)(params_traj))
end = perf_counter()
print(f"the time it took for trajectory loss calculation {end - start}")



def accuracy(model: nets.nn.Module, params, batch):
    logits = model.apply(params, batch['image'])
    accuracy = nets.compute_accuracy(logits, batch['label'])
    return accuracy

G, gradG = nets.G_function(params_traj, model, G_label, G_Temp, G_l1reg, G_tvreg)

state_x = key, MNIST[1]['image']
hypsG = G, gradG, G_eta, 0., 1.

start = perf_counter()
(last_key, last_image), x_traj = langevin.MALA_chain(state_x, hypsG, G_Steps)

import plotting

plotting.show_mnist_example(x_traj[G_Steps-1])
end = perf_counter()
print(f"calculating the G trajectory takes {end - start}")