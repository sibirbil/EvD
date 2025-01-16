
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import create_datasets
import model_contrast_functions

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

import nets, train, optax, utils

import logistic, langevin
import matplotlib.pyplot as plt
import seaborn as sns



df, X, y, scaler_X, scaler_y = create_datasets.get_housing()
key = random.key(42)
init_key, dropout_key, train_key, mala_key, x0_key = random.split(key, 5)

train_idx = random.choice(key, jnp.arange(len(X)), shape=(int(0.8 * len(X)),), replace=False)
test_idx = jnp.delete(jnp.arange(len(X)), train_idx)
X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]


# SVR
# Hyperparameters are calculated with grid search
svr_model = SVR(kernel='linear', C=10, epsilon=0.01)
svr_model.fit(X_train, y_train)

# Make predictions
svr_train_preds = svr_model.predict(X_train)
svr_test_preds = svr_model.predict(X_test)

# Evaluate the model
svr_train_mse = mean_squared_error(y_train, svr_train_preds)
svr_test_mse = mean_squared_error(y_test, svr_test_preds)
svr_train_r2 = r2_score(y_train, svr_train_preds)
svr_test_r2 = r2_score(y_test, svr_test_preds)

# Print results
print("Support Vector Regressor:")
print(f"Train MSE: {svr_train_mse:.4f}")
print(f"Test MSE: {svr_test_mse:.4f}")
print(f"Train R2: {svr_train_r2:.4f}")
print(f"Test R2: {svr_test_r2:.4f}")

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_train_preds = lin_reg.predict(X_train)
lin_test_preds = lin_reg.predict(X_test)

# Metrics
print("\nLinear Regression:")
print(f"Train MSE: {mean_squared_error(y_train, lin_train_preds):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, lin_test_preds):.4f}")
print(f"Train R2: {r2_score(y_train, lin_train_preds):.4f}")
print(f"Test R2: {r2_score(y_test, lin_test_preds):.4f}")



beta = 1000.
eta = 0.01/beta
G_sml = model_contrast_functions.G_similar_function(lin_reg, svr_model, beta)
gradG_sml = jax.grad(G_sml)

hypsG_sml = G_sml, gradG_sml, eta, 0.0, 1.0
x0 = jax.random.uniform(x0_key, X.shape[1])
state_x = mala_key, x0

_, traj_x_sml = langevin.MALA_chain(state_x, hypsG_sml, 5000)
synt_data_sml = traj_x_sml[-500:]
linear_preds_sml = lin_reg.predict(synt_data_sml)
svr_preds_sml = svr_model.predict(synt_data_sml)
inverted_ln_y_sml = scaler_y.inverse_transform(np.reshape(linear_preds_sml, (-1,1)))
inverted_svr_y_sml = scaler_y.inverse_transform(np.reshape(svr_preds_sml, (-1,1)))


G_cnt = model_contrast_functions.G_contrast_function(lin_reg, svr_model, beta)
gradG_cnt = jax.grad(G_cnt)

hypsG_cnt = G_cnt, gradG_cnt, eta, 0.0, 1.0
x0 = jax.random.uniform(x0_key, X.shape[1])
state_x = mala_key, x0

_, traj_x_cnt = langevin.MALA_chain(state_x, hypsG_cnt, 5000)
synt_data_cnt = traj_x_cnt[-500:]
inverted_synt_data = scaler_X.inverse_transform(synt_data_cnt)


## Check the maximum and minimum values in the synthetic data and compare it with the original data
model_contrast_functions.print_max_min_values(inverted_synt_data, df.columns[1:], "Synthetic Data")
model_contrast_functions.print_max_min_values(df.iloc[:, 1:].to_numpy(), df.columns[1:], "Original Data")


linear_preds_cnt = lin_reg.predict(synt_data_cnt)
svr_preds_cnt = svr_model.predict(synt_data_cnt)
inverted_ln_y_cnt = scaler_y.inverse_transform(np.reshape(linear_preds_cnt, (-1,1)))
inverted_svr_y_cnt = scaler_y.inverse_transform(np.reshape(svr_preds_cnt, (-1,1)))


##### Visualisations ########

# Scatter plot of Linear Regression vs SVR predictions
model_contrast_functions.scatter_plot_with_reference(
    x=linear_preds_cnt,
    y=svr_preds_cnt,
    x_label="Linear Regression Predictions",
    y_label="SVR Predictions",
    title="Scatter Plot of Linear Regression vs SVR Predictions for Model Contrast",
    color="blue",
    alpha=0.6
) 

# Box plot to compare features
model_contrast_functions.side_by_side_boxplots(
    data1=synt_data_cnt,
    data2=X_test,    # or synt_data_sml?
    labels=df.columns[1:],
    title="Feature Comparison Between Datasets",
    x_label="Features",
    y_label="Values"
)


# Scatter plot of Linear Regression vs SVR predictions on the Test Data
model_contrast_functions.scatter_plot_with_reference(
    x=lin_test_preds,
    y=svr_test_preds,
    x_label="Linear Regression Predictions",
    y_label="SVR Predictions",
    title="Scatter Plot of Linear Regression vs SVR Predictions on the Test Data",
    color="blue",
    alpha=0.6
)
