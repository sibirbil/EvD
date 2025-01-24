
import logistic
import langevin
import pandas as pd

import jax
import jax.random as random
import jax.numpy as jnp
from datasets import Dataset
import model_contrast_functions

from sklearn.linear_model import LogisticRegression
import create_datasets
from functools import partial
import nets, train, optax, utils
import matplotlib.pyplot as plt


fico_df, X, y, scaler = create_datasets.get_fico()
key = random.key(42)
init_key, train_key, mala_key, x0_key = random.split(key,4)
train_idx = random.choice(key, jnp.arange(len(X)), shape = (int(0.9*len(X)),),  replace=False)
test_idx = jnp.delete(jnp.arange(len(X)), train_idx)
X_train = jnp.array(X[train_idx])
y_train = jnp.array(y[train_idx])
X_test = jnp.array(X[test_idx])
y_test = jnp.array(y[test_idx])


linear = nets.MLP(features=[1])

betaF = 1000.
etaF = 0.1/betaF
N_params = 10000

log_reg = LogisticRegression(max_iter = 15000)
log_reg.fit(X_train, y_train)
w, b = log_reg.coef_, log_reg.intercept_
params0 = logistic.create_params_from_array(w.T,b)

params_state = key, params0
F = logistic.F_function(X_train, y_train, linear, betaF)
hypsF = F, jax.grad(F), etaF
_, traj_params = langevin.MALA_chain(params_state, hypsF, N_params)


betaG= 1000.
etaG = 0.01/betaG
etaG = utils.sqrt_decay(etaG)
N_x = 10000


x0 = X[0]
state_x = key, x0

G = logistic.G_function(traj_params, linear, logistic.constant_estimator(0.5),
                        logistic.square_cost, partial(logistic.l2_reg, C = 0., x0=x0), 
                        betaG)
gradG = jax.grad(G)

hypsG = G, gradG, etaG, 0.0, 1.0

_, traj_x = langevin.MALA_chain(state_x, hypsG, N_x)

inverted = scaler.inverse_transform(traj_x)


original_data = fico_df.drop(fico_df.columns[0], axis=1)

ys = log_reg.predict(traj_x)
# Check predictions
y_prob_check = log_reg.predict_proba(traj_x)[:, 1] 

data_path = jnp.column_stack([y_prob_check, traj_x])
data_path_df = pd.DataFrame(data_path, columns= fico_df.columns)

# Adjust columns for the inverted dataset
inverted_df = pd.DataFrame(inverted[-500:], columns=original_data.columns)

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(y_prob_check[-500:], bins=20, edgecolor='black', color='skyblue', alpha=0.7)
plt.axvline(0.5, color='red', linestyle='dashed', linewidth=1, label='0.5')

plt.xlabel('Prediction Probabilities')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Probabilities Around 0.5')
plt.legend()
plt.show()

## Check the maximum and minimum values in the synthetic data and compare it with the original data
model_contrast_functions.print_max_min_values(inverted_df.to_numpy(), original_data.columns, "Synthetic Data")
model_contrast_functions.print_max_min_values(original_data.to_numpy(), original_data.columns, "Original Data")


# Compare datasets
model_contrast_functions.compare_datasets(original=original_data, inverted=inverted_df)

