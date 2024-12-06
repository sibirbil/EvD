import create_datasets
import nets
import jax
import jax.random as random
import jax.numpy as jnp

from functools import partial


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import logistic
import langevin


key = random.PRNGKey(1826)  # yeniceri ocaginin kapatilmasi ya da ilk sadece matematik dergisi CRELLE's journal'in kurulmasi

adult, _, _  = create_datasets.get_adult()  # check the output signature of the function if it stops working

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
etaG = 0.001/betaG
N_x = 1000

x0 = X_train[0]
state_x = key, x0

G = logistic.G_function(traj_params, linear, logistic.constant_estimator(1.), logistic.cross_entropy, partial(logistic.l1_reg, x0 = x0), betaG)
hypsG = G, jax.grad(G), etaG

_, traj_x = langevin.MALA_chain(state_x, hypsG, N_x)

a = jax.vmap(G)(traj_x)/betaG
print(f"x trajectory quality max: {jnp.max(a)}, min:{jnp.min(a)}, mean:{jnp.mean(a)}, std:{jnp.std(a)}")

