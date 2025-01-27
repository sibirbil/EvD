import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pandas as pd
from functools import partial
import flax.linen as nn
from xgboost import XGBClassifier

import create_datasets
import nets
import train
import datasets
import langevin
import vi_opt


df, X, y, scaler = create_datasets.get_fico()

N = X.shape[0]
key = random.key(24)
init_key, train_key, mala_key, noise_key, x0_key = random.split(key, 5)


## Get the data in JAX format

train_idx = random.choice(key, jnp.arange(N), shape = (int(0.9*N),),  replace=False)
test_idx = jnp.delete(jnp.arange(N), train_idx)
X_train = jnp.array(X)[train_idx]
y_train = jnp.array(y)[train_idx]
X_test = jnp.array(X)[test_idx]
y_test = jnp.array(y)[test_idx]

ds_train = datasets.Dataset.from_dict({'x':X_train, 'y':y_train})
ds_test = datasets.Dataset.from_dict({'x':X_test, 'y':y_test})
ds_train = ds_train.with_format('jax')
ds_test = ds_test.with_format('jax')


## Define the MLP network with dropout and train it.

mlp = nets.MLP_with_dropout(features = [128,32,8,2], dropout_rate=0.2)
exp_decay_schedule = optax.schedules.exponential_decay(0.01,100, 0.9,500, True, end_value = 1e-5)
adam = optax.adam(learning_rate = exp_decay_schedule)
ts = train.create_train_state(init_key, mlp, X_train[0].shape, adam)
ts = train.train(train_key,ts, ds_train, 128, 10000)
test_acc, test_loss = train.eval_step(ts, ds_test[:])

print(f"The accuracy on the test set {test_acc} with loss {test_loss}")

# we add a bit of noise to the parameters
N_noise = 1000
noise_std = 0.01
noised_params = vi_opt.gaussian_single_std_samples(noise_key, ts.params, noise_std, N_noise)

## score of a single set of params on a given dataset
def score(params, X, y):
    logits = mlp.apply(params, X, is_training = False)
    preds = jnp.argmax(logits, axis = 1)
    return (preds == y).mean()




## This defines the loss function on the parameters for using MALA.

def F_function(
    X       : jax.Array,
    y       : jax.Array,  #integer labels
    model   : nn.Module,
    beta    : jnp.float_
    ):
    def F(params):
        logits = model.apply(params, X, is_training= False)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return beta*jnp.mean(losses)
    return jax.jit(F)


betaF = 1000.
etaF = 0.1/betaF
N_params = 10
params_state = mala_key, ts.params
F = F_function(ds_train['x'], ds_train['y'], mlp, betaF)
hypsF = F, jax.grad(F), etaF
_, traj_params = langevin.MALA_chain(params_state, hypsF, N_params)


def G_function_risky(
    traj_params,    # a single PyTree
    target          : jax.Array,    # target softmax values of shape (k,)
    model           : nn.Module,
    beta            : jnp.float_    # inverse temperature
):
    model_fn = partial(model.apply, is_training =False)
    def G(x):
        logits = jax.vmap(model_fn, in_axes = (0, None))(traj_params, x)
        losses = -jax.nn.log_softmax(logits) @ target
        return beta*jnp.mean(losses)
    
    return jax.jit(G)

def G_function_sensitive(
    ref_params,     #the undisturbed params
    noise_params,    #single pytree
    model   : nn.Module,
    beta    : jnp.float_
    ):
    """
    This G-function prefers those labels 
    """
    model_fn = partial(model.apply, is_training = False)
    def G(x):
        logits = jax.vmap(model_fn, in_axes = (0, None))(noise_params, x)
        label = jnp.argmax(model_fn(ref_params, x), axis = -1)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits.squeeze(), 1-label)
        return beta*jnp.mean(losses)
    
    return jax.jit(G)


clsfr = XGBClassifier(
    subsample=0.6, 
    reg_lambda=5, 
    reg_alpha=10, 
    n_estimators=200, 
    min_child_weight=1, 
    max_depth=8, 
    learning_rate=0.05, 
    gamma=1, 
    colsample_bytetree=0.5
    )
clsfr.fit(X_train, y_train)
print(f"The xgboost classifier has score \
      {clsfr.score(X_train, y_train)} on training set \
      and {clsfr.score(X_test, y_test)} on test set.")

betaG = 100.
etaG = 0.01/betaG
N_data = 20000

#x0 = jax.random.uniform(x0_key, shape = X_train[0].shape)
x0 = X_train[0]
x_state = x0_key, x0
G = G_function_sensitive(ts.params, noised_params, mlp, betaG)
hypsG = G, jax.grad(G), etaG, 0., 1.
#Generate data points which are sensitive to noise
_, traj_x = langevin.MALA_chain(x_state, hypsG, N_data)

# see their predictions under original parameters
traj_x_thetastar_logits = mlp.apply(ts.params, traj_x, is_training = False)
traj_x_thetastar_probs = jax.nn.softmax(traj_x_thetastar_logits)
traj_x_thetastar_preds = jnp.argmax(traj_x_thetastar_probs, axis = -1)  # shape (N_data,)
# see their predictions under noised parameters
traj_x_noised_logits = jax.vmap(mlp.apply, in_axes = (0, None, None))(noised_params, traj_x, False)
traj_x_noised_probs = jax.nn.softmax(traj_x_noised_logits)
traj_x_noised_preds = jnp.argmax(traj_x_noised_logits, axis = -1)   # shape (N_noise, N_data)

sameornot = (traj_x_noised_preds == traj_x_thetastar_preds) # shape (N_Noise, N_data)
# persistence of predictions
persistence = jnp.mean(sameornot, axis = 0)
print(f"The percent of persistence {persistence[-100:]}")



# compare the two predictions
noised_scores = jax.vmap(score, in_axes = (0,None, None))(noised_params, traj_x, traj_x_thetastar_preds)

inverted_noised_traj = scaler.inverse_transform(traj_x)
generated_noised_df = pd.DataFrame(jnp.column_stack([traj_x_thetastar_preds, inverted_noised_traj]), columns = df.columns)

test_logits = mlp.apply(ts.params, X_test, is_training =False)
test_probs = jax.nn.softmax(test_logits)
test_preds = jnp.argmax(test_probs, axis = -1)

# measuring how robust the test_set predictions are to adding of noise.
test_noised_logits = jax.vmap(mlp.apply, in_axes = (0, None, None))(noised_params, X_test, False)
test_noised_preds = jnp.argmax(test_noised_logits, axis = -1)
test_persistence = jnp.mean((test_noised_preds == test_preds), axis = 0)  #shape (N_test,)

G_risky = G_function_risky(traj_params, jnp.array([1,1]), mlp, betaG)
hypsG_risky = G_risky, jax.grad(G_risky), etaG
_, risky_traj_x = langevin.MALA_chain(x_state, hypsG_risky, N_data)


risky_traj_logits = mlp.apply(ts.params, risky_traj_x, is_training = False)
risky_traj_probs = jax.nn.softmax(risky_traj_logits)
risky_traj_preds = jnp.argmax(risky_traj_probs, axis = -1)

inverted_risky_traj = scaler.inverse_transform(risky_traj_x)
generated_risky_df = pd.DataFrame(jnp.column_stack([risky_traj_preds, inverted_risky_traj]), columns = df.columns)

test_df = pd.DataFrame(jnp.column_stack([y_test, scaler.inverse_transform(X_test)]), columns = df.columns)
test0_df = test_df[test_df['RiskPerformance']==0]
test1_df = test_df[test_df['RiskPerformance']==1]

noise_traj_prob_std = jnp.std(traj_x_thetastar_probs[-N_data//2:, 0])
risky_traj_prob_std = jnp.std(risky_traj_probs[-N_data//2:, 0])
print(f"the std of noise {noise_traj_prob_std} and risky samples are {risky_traj_prob_std}")

import plotting

figure = plotting.feature_comparison_histograms([generated_noised_df, generated_risky_df, test0_df, test1_df],
                                                labels = ['generated_noised', 'generated_risky', 'test_with_label0', 'test_with_label1'])
figure.show()