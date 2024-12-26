from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import create_datasets
from datasets import Dataset

import jax
import jax.random as random
import jax.numpy as jnp

import nets, train, optax, utils

import logistic, langevin

df, X, y, scaler = create_datasets.get_fico()
key = random.key(42)
init_key, dropout_key, train_key, mala_key, x0_key = random.split(key,5)
train_idx = random.choice(key, jnp.arange(len(X)), shape = (int(0.9*len(X)),),  replace=False)
test_idx = jnp.delete(jnp.arange(len(X)), train_idx)
X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]
ds = Dataset.from_dict({'x':X_train, 'y':y_train})
ds_test = Dataset.from_dict({'x':X_test, 'y':y_test})
ds.set_format('jax')
ds_test.set_format('jax')

btc = XGBClassifier() #boosted tree classifier, I don't know about options
btc.fit(ds['x'], ds['y'])
print(f"xgboost: \t train acc {btc.score(X_train,y_train)}, \t test acc {btc.score(X_test, y_test)}")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(f"log_reg: \t train acc {log_reg.score(X_train, y_train)} \t test acc {log_reg.score(X_test, y_test)}")



# The below function can be written via the logistic.G_function function factory.
# but I realized this after I wrote it, and I don't want to change it now.


def G_contrast_function(log_reg, btc, beta):
    w = jnp.array(log_reg.coef_)
    b = jnp.array(log_reg.intercept_)
    linear_params = logistic.create_params_from_array(w.T,b)
    linear = nets.MLP([1])
    
    def G(x):
        logit = linear.apply(linear_params, x)
        x_copy = jax.lax.stop_gradient(x) #gradient is not computed through here
        print(type(x))
        y_compare = jnp.array(btc.predict(x_copy[None, :]))
        return beta*logistic.cross_entropy(logit, y_compare)
    
    return G

# beta = 1000.
# eta = 0.01/beta
# G_tilde = G_contrast_function(log_reg, btc, beta)
# G = lambda z: G_tilde(jax.nn.sigmoid(4*z-2.))
# gradG = jax.grad(G)
# hypsG = G, gradG, eta
# x0 = jax.random.uniform(x0_key, X.shape[1])
# state_x = mala_key, x0

#####
# These lines below do not work because of a stupid reason, because G uses xgboost.predict 
# and that uses numpy. But we need everything to be in jax.numpy for jax.lax.scan in MALA to work.
# I may find a way around it, maybe use another library, but I'm letting it go for now.
# one way would be to modify MALA so that it doesn't use jax.lax.scan and we do a simple loop.

#_, traj_x = langevin.MALA_chain(state_x, hypsG, 1000)
#logistic_preds = log_reg.predict(traj_x)
#btc_preds = btc.predict(traj_x)


mlp = nets.MLP_with_dropout(features = [128, 32, 8, 2], dropout_rate = 0.2)
params_mlp = mlp.init(init_key, ds[:2]['x'])
learning_rate = utils.sqrt_decay(0.01)
momentum = 0.9
optimizer = optax.sgd(learning_rate, momentum)
mts = train.TrainState.create(apply_fn = mlp.apply, params = params_mlp, rng_key = dropout_key, tx = optimizer)

mts = train.train(train_key, mts, ds, 64, 10001)

def predict(model, params, X):
    return jnp.argmax(model.apply(params, X, is_training = False), axis = 1)

test_acc, test_loss = train.eval_step(mts, ds_test[:])
print(f"for mlp the test acc:{test_acc} and loss: {test_loss}")

def G_contrast_function2(log_reg, model, params, beta):
    w = jnp.array(log_reg.coef_)
    b = jnp.array(log_reg.intercept_)
    linear_params = logistic.create_params_from_array(w.T,b)
    linear = nets.MLP([1])
    
    def G(x):
        logit = linear.apply(linear_params, x)
        #x_copy = jax.lax.stop_gradient(x) #gradient is not computed through here
        y_compare = jnp.array(predict(model, params, x))
        return beta*logistic.cross_entropy(logit, 1-y_compare)
    
    return G

beta = 1000.
eta = 0.01/beta
G_tilde = G_contrast_function2(log_reg, mlp, mts.params, beta)
G = lambda z: G_tilde(jax.nn.sigmoid(4*z-2.))
gradG = jax.grad(G)
hypsG = G, gradG, eta
x0 = jax.random.uniform(x0_key, X.shape[1])
state_x = mala_key, x0

from functools import partial

_, traj_z = langevin.MALA_chain(state_x, hypsG, 1000)
traj_x = jax.nn.sigmoid(4*traj_z -2.)
logistic_preds = log_reg.predict(traj_x)
mlp_preds = jax.vmap(partial(predict,mlp, mts.params))(traj_x)

data = scaler.inverse_transform(traj_x)
colnames = df.drop(columns = "RiskPerformance").columns
import pandas as pd

#The disagreeing data
traj_readable = pd.DataFrame(data, columns = colnames)