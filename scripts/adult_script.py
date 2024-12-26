import create_datasets
import nets
import jax
import jax.random as random
import jax.numpy as jnp
import pandas as pd
import numpy as np

from functools import partial

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import logistic
import langevin
import utils


key = random.PRNGKey(1826)  # yeniceri ocaginin kapatilmasi ya da ilk sadece matematik dergisi CRELLE's journal'in kurulmasi

adult, preprocessor, df  = create_datasets.get_adult()  # check the output signature of the function if it stops working

adult_train, adult_test = train_test_split(adult, test_size = 0.1)

X_train = jnp.array(adult_train.drop(columns = 'income').to_numpy())
y_train = jnp.array(adult_train['income'].to_numpy())

X_test = jnp.array(adult_test.drop(columns = 'income').to_numpy())
y_test = jnp.array(adult_test['income'].to_numpy())

# Write the linnear model as a 1 layer single output neural network.
linear = nets.MLP(features=[1], input_size  =  len(adult.columns) - 1)

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

a = jax.vmap(F)(traj_params)/betaF
print(f"parameter trajectory quality max: {jnp.max(a)}, min:{jnp.min(a)}, mean:{jnp.mean(a)}, std:{jnp.std(a)}")

betaG= 1000.
etaG = 0.01/betaG
etaG = utils.sqrt_decay(etaG)
N_x = 1000

x0 = X_train[1]
state_x = key, x0


neg_predictor = logistic.negation_logistic_estimator(params0, linear)


#G = logistic.G_function(traj_params, linear, logistic.constant_estimator(1.), logistic.cross_entropy, partial(logistic.l1_reg, C = 0., x0 = x0), betaG)
G = logistic.G_function(traj_params, linear, logistic.constant_estimator(0.5), logistic.square_cost, partial(logistic.l1_reg, C = 0., x0 = x0), betaG)
#G = logistic.G_function(traj_params, linear, neg_predictor, logistic.cross_entropy, partial(logistic.l2_reg, C = 0., x0 = x0), betaG)
#G = logistic.G_function(traj_params, linear, neg_predictor, logistic.cross_entropy, partial(logistic.l1_reg, C = 1., x0 = x0), betaG)
hypsG = G, jax.grad(G), etaG, 0., 1.

_, traj_x = langevin.MALA_chain(state_x, hypsG, N_x)

a = jax.vmap(G)(traj_x)/betaG
print(f"x trajectory quality max: {jnp.max(a)}, min:{jnp.min(a)}, mean:{jnp.mean(a)}, std:{jnp.std(a)}")

ys = log_reg.predict(traj_x)
data_path = jnp.column_stack([traj_x, ys])
data_path_df = pd.DataFrame(data_path, columns= adult.columns)
inverted = create_datasets.invert_adult(data_path_df, preprocessor)

