from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import create_datasets, logistic, langevin, nets, train, utils

from datasets import Dataset
import pandas as pd

import jax
import jax.random as random
import jax.numpy as jnp


df, X, y, scaler = create_datasets.get_fico()
N = X.shape[0]
key = random.key(42)
init_key, dropout_key, train_key, mala_key, x0_key = random.split(key,5)
train_idx = random.choice(key, jnp.arange(N), shape = (int(0.9*N),),  replace=False)
test_idx = jnp.delete(jnp.arange(N), train_idx)
X_train = jnp.array(X)[train_idx]
y_train = jnp.array(y)[train_idx]
X_test = jnp.array(X)[test_idx]
y_test = jnp.array(y)[test_idx]
test_df = pd.DataFrame(jnp.column_stack([y_test, scaler.inverse_transform(X_test)]), columns = df.columns)
ds = Dataset.from_dict({'x':X_train, 'y':y_train})
ds_test = Dataset.from_dict({'x':X_test, 'y':y_test})
ds.set_format('jax')
ds_test.set_format('jax')


xgbclsfr = XGBClassifier(
    subsample=0.6, 
    reg_lambda=5, 
    reg_alpha=10, 
    n_estimators=200, 
    min_child_weight=1, 
    max_depth=8, 
    learning_rate=0.05, 
    gamma=1, 
    colsample_bytree=0.5
    )
xgbclsfr.fit(X_train, y_train)
print(f"xgboost: \t train acc {xgbclsfr.score(X_train,y_train)}, \t test acc {xgbclsfr.score(X_test, y_test)}")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(f"log_reg: \t train acc {log_reg.score(X_train, y_train)} \t test acc {log_reg.score(X_test, y_test)}")



# The below function can be written via the logistic.G_function function factory.
# but I realized this after I wrote it, and I don't want to change it now.


def G_contrast_function(log_reg_mdl :LogisticRegression, xgb_mdl :XGBClassifier, beta):
    w = jnp.array(log_reg_mdl.coef_)
    b = jnp.array(log_reg_mdl.intercept_)
    linear_params = logistic.create_params_from_array2(w,b)
    linear = nets.MLP_with_dropout([1], dropout_rate=0.)
    
    def G(x):
        logit = linear.apply(linear_params, x, is_training = False)
        x_copy = jax.lax.stop_gradient(x) #gradient is not computed through here
        y_compare = jnp.array(xgb_mdl.predict(x_copy[None, :]))
        return beta*logistic.cross_entropy(logit, 1-y_compare)
    
    return G

beta = 1000.
eta = 0.01/beta
G_tilde = G_contrast_function(log_reg, xgbclsfr, beta)
G = lambda z: G_tilde(jax.nn.sigmoid(4*z-2.))
gradG = jax.grad(G)
hypsG = G, gradG, eta
z0 = jax.random.uniform(x0_key, (X.shape[1],))
state_z = mala_key, z0


_, traj_z = langevin.nt_MALA(state_z, hypsG, 2000)
traj_x = jax.nn.sigmoid(4*traj_z - 1)
logistic_preds = log_reg.predict(traj_x)
xgb_preds = xgbclsfr.predict(traj_x)


generated_x = scaler.inverse_transform(traj_x)
generated_df = pd.DataFrame(jnp.column_stack([xgb_preds, generated_x]), columns=df.columns)
generated_df = generated_df[generated_df['RiskPerformance']==1] #making sure all 

if len(generated_df) <500:
    print("not enough data points generated to make graphs")

import model_contrast_functions

## Prints all the images one by one

# model_contrast_functions.compare_datasets(
#     data1=generated_df[-500:],
#     data2=test_df[-500:],
#     labels=('Generated Contrastive Data', 'Test Data')
# )

log_reg_test_pred = log_reg.predict(X_test)
xgboost_test_pred = xgbclsfr.predict(X_test)
print(f"On the test data predictions of XGBoost and logistic regression agree \
      {(log_reg_test_pred == xgboost_test_pred).mean():.2%} ")

model_contrast_functions.compare_datasets_grid(
    data1=generated_df[-500:],
    data2=test_df[-500:],
    numerical_cols=['MaxDelqEver', 'NumTotalTrades', 'MSinceMostRecentTradeOpen', 'NumTradesOpeninLast12M'],
    labels=('Generated Data', 'Test Data')
    )


# mlp = nets.MLP_with_dropout(features = [128, 32, 8, 2], dropout_rate = 0.2)
# N_train = 10_000

# ts = mlp_fico_train.train_if_exists(train_key, mlp, X_train, y_train, N_train)
# params_mlp = mlp.init(init_key, ds[:2]['x'])
#learning_rate = utils.sqrt_decay(0.01)
#momentum = 0.9
#optimizer = optax.sgd(learning_rate, momentum)
#mts = train.TrainState.create(apply_fn = mlp.apply, params = params_mlp, rng_key = dropout_key, tx = optimizer)

#mts = train.train(train_key, mts, ds, 64, 10001)

# def predict(model, params, X):
#     return jnp.argmax(model.apply(params, X, is_training = False), axis = 1)

# test_acc, test_loss = train.eval_step(ts, ds_test[:])
# print(f"for mlp the test acc:{test_acc} and loss: {test_loss}")

# def G_contrast_function2(log_reg, model, params, beta):
#     w = jnp.array(log_reg.coef_)
#     b = jnp.array(log_reg.intercept_)
#     linear_params = logistic.create_params_from_array(w.T,b)
#     linear = nets.MLP([1])
    
#     def G(x):
#         logit = linear.apply(linear_params, x)
#         #x_copy = jax.lax.stop_gradient(x) #gradient is not computed through here
#         y_compare = jnp.array(predict(model, params, x))
#         return beta*logistic.cross_entropy(logit, 1-y_compare)
    
#     return G

# beta = 1000.
# eta = 0.01/beta
# G_tilde = G_contrast_function2(log_reg, mlp, ts.params, beta)
# G = lambda z: G_tilde(jax.nn.sigmoid(4*z-2.))
# gradG = jax.grad(G)
# hypsG = G, gradG, eta
# x0 = jax.random.uniform(x0_key, X.shape[1])
# state_x = mala_key, x0

# # _, traj_z = langevin.MALA_chain(state_x, hypsG, 1000)
# # traj_x = jax.nn.sigmoid(4*traj_z -2.)
# # logistic_preds = log_reg.predict(traj_x)
# # mlp_preds = jax.vmap(partial(predict,mlp, mts.params))(traj_x)

# # data = scaler.inverse_transform(traj_x)
# # colnames = df.drop(columns = "RiskPerformance").columns
# # import pandas as pd

# # #The disagreeing data
# # traj_readable = pd.DataFrame(data, columns = colnames)