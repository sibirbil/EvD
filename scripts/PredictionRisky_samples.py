
import logistic
import langevin
import pandas as pd
import flax.linen as nn
import numpy

import jax
import jax.random as random
import jax.numpy as jnp
import datasets
import model_contrast_functions

from sklearn.linear_model import LogisticRegression
import create_datasets
from functools import partial
import nets, train, optax, utils
import matplotlib.pyplot as plt
import model_contrast_functions

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
N_params = 10000
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


betaG= 1000.
etaG = 0.01/betaG
etaG = utils.sqrt_decay(etaG)
N_data = 20000

# Columns without the label column (assumed first)
original_data = df.drop(df.columns[0], axis=1)

G_risky = G_function_risky(traj_params, jnp.array([1.0, 1.0]), mlp, betaG)
hypsG_risky = G_risky, jax.grad(G_risky), etaG, 0., 1.
x0 = X_train[0]

key_list = random.split(x0_key, 50)
all_inverted_runs = []   # list of DataFrames (each last 500 samples, inverse-scaled)
all_y_prob_runs = []     # list of arrays of p(class=1) for the last 500 samples


for key_i in key_list:
    x_state_i = key_i, x0
    _, traj_i = langevin.MALA_chain(x_state_i, hypsG_risky, N_data)
    # model probabilities for this run
    risky_traj_logits_i = mlp.apply(ts.params, traj_i, is_training=False)
    risky_traj_probs_i = jax.nn.softmax(risky_traj_logits_i)

    # take last 500
    generated_sample_i = traj_i[-500:, ]
    y_prob_i = risky_traj_probs_i[-500:, 1]  # probability of positive class

    # inverse transform last 500
    inverted_i = scaler.inverse_transform(generated_sample_i)
    # columns without the (first) label column
    original_data = df.drop(df.columns[0], axis=1)
    inverted_df_i = pd.DataFrame(inverted_i, columns=original_data.columns)

    all_inverted_runs.append(inverted_df_i)
    all_y_prob_runs.append(numpy.array(y_prob_i))
    
    
##x0 = jax.random.uniform(x0_key, shape = X_train[0].shape)
#x_state = x0_key, x0
#_, risky_traj_x = langevin.MALA_chain(x_state, hypsG_risky, N_data)
#risky_traj_logits = mlp.apply(ts.params, risky_traj_x, is_training = False)
#risky_traj_probs = jax.nn.softmax(risky_traj_logits)
#risky_traj_preds = jnp.argmax(risky_traj_probs, axis = -1)

#generated_sample = risky_traj_x[-500:,]
#inverted = scaler.inverse_transform(generated_sample)


#original_data = df.drop(df.columns[0], axis=1)


#data_path = jnp.column_stack([risky_traj_probs[:, 1] , risky_traj_x])
#data_path_df = pd.DataFrame(data_path, columns = df.columns)

# Adjust columns for the inverted dataset
#inverted_df = pd.DataFrame(inverted, columns=original_data.columns)
#y_prob = risky_traj_probs[-500:, 1] 


#mean_y_prob = numpy.mean(y_prob)       
#std_y_prob = numpy.std(y_prob)  

# ---------------------------
# Aggregate simple stats across runs (optional; keep your plotting separate)
# ---------------------------
y_prob_all = numpy.concatenate(all_y_prob_runs, axis=0)
mean_y_prob = numpy.mean(y_prob_all)
std_y_prob = numpy.std(y_prob_all)
print(f"Across 30 runs — mean p(class=1) over last 500 samples/run: {mean_y_prob:.4f}")
print(f"Across 30 runs — std  p(class=1) over last 500 samples/run: {std_y_prob:.4f}")

# Histogram (aggregated across runs) — you can keep or remove
plt.figure(figsize=(10, 6))
plt.hist(y_prob_all, bins=20, edgecolor='black', color='skyblue', alpha=0.7)
plt.axvline(0.5, color='red', linestyle='dashed', linewidth=1, label='0.5')
plt.xlabel('Prediction Probabilities')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Probabilities (Aggregated over 30 runs)')
plt.legend()
plt.show()


# ---------------------------
# Max/min & dataset comparisons (use concatenated synthetic set)
# ---------------------------
synthetic_concat = pd.concat(all_inverted_runs, ignore_index=True)
model_contrast_functions.print_max_min_values(
    synthetic_concat.to_numpy(), original_data.columns, "Synthetic Data (30 runs, concatenated)"
)
model_contrast_functions.print_max_min_values(
    original_data.to_numpy(), original_data.columns, "Original Data"
)

# Compare datasets (using concatenated synthetic for now)
model_contrast_functions.compare_datasets_with_error(
    factual=original_data,
    all_runs=all_inverted_runs,
    cols=4,
    max_plots=8,           # keep the old 2x4 look
    bins=60,
    bar_width=0.35,
    xtick_fontsize=10
)

model_contrast_functions.compare_datasets_gridAll_with_error(
    original_data=original_data,
    all_runs=all_inverted_runs,
    labels=('Original Data', 'Generated (mean ± std)'),
    max_features=20,          # 5x4 grid
    bins=60,
    bar_width=0.4,
    xtick_fontsize=10
)


