import logistic
import jax
import jax.numpy as jnp
import langevin
import nets

from sklearn.linear_model import LogisticRegression
import create_datasets
import utils
import vi_opt
from functools import partial

gmsc_df, X, y, scaler = create_datasets.get_gmsc()
key = jax.random.PRNGKey(1986)

X = jnp.array(X)
y = jnp.array(y.to_numpy())

linear = nets.MLP(features = [1], input_size = 10)

log_reg = LogisticRegression(max_iter=15000)
log_reg.fit(X,y)

betaF = 1000.
etaF = 0.1/betaF
N_params = 10000

log_reg = LogisticRegression(max_iter = 15000)
log_reg.fit(X, y)
w, b = log_reg.coef_, log_reg.intercept_
params0 = logistic.create_params_from_array(w.T,b)

params_state = key, params0
F = logistic.F_function(X, y, linear, betaF)
hypsF = F, jax.grad(F), etaF
_, traj_params = langevin.MALA_chain(params_state, hypsF, N_params)

betaG= 1000.
etaG = 0.01/betaG
etaG = utils.sqrt_decay(etaG)
N_x = 1000
sigma_x = 0.1

x0 = X[0]
state_x = key, x0

G_tilde = logistic.G_function(traj_params, linear, logistic.constant_estimator(1.), logistic.cross_entropy, partial(logistic.l2_reg, C = 0., x0=x0), betaG)

G = lambda z: G_tilde(jax.nn.sigmoid(4*z-2.))  # this is near identity for most of the [0,1] interval but also keeps the values in [0, 1]
gradG = jax.grad(G)

hypsG = vi_opt.Hyps(G, gradG, etaG, nSamples=3, scale= sigma_x)

traj_z = vi_opt.vi_traj(state_x, hypsG, N_x)
traj_x = jax.nn.sigmoid(4*traj_z - 2.)

inverted = scaler.inverse_transform(traj_x)