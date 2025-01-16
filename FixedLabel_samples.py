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
import summarize_results


key = random.PRNGKey(1826)  # yeniceri ocaginin kapatilmasi ya da ilk sadece matematik dergisi CRELLE's journal'in kurulmasi
np.random.seed(44)  # NumPy's random seed


adult, preprocessor, df  = create_datasets.get_adult()  # check the output signature of the function if it stops working

adult_train, adult_test = train_test_split(adult, test_size = 0.1, random_state= 1826)

# Reset the index of the training dataset
adult_train.reset_index(drop=False, inplace=True)  # Creates a new index column based on row positions
adult_train = adult_train.drop(columns=['index'])

X_train = jnp.array(adult_train.drop(columns = 'income').to_numpy())
y_train = jnp.array(adult_train['income'].to_numpy())

X_test = jnp.array(adult_test.drop(columns = 'income').to_numpy())
y_test = jnp.array(adult_test['income'].to_numpy())

# Write the linnear model as a 1 layer single output neural network.
linear = nets.MLP(features=[1])

betaF = 1000.
etaF = 0.1/betaF
N_params = 10000

log_reg = LogisticRegression(max_iter = 15000)
log_reg.fit(X_train, y_train)
w, b = log_reg.coef_, log_reg.intercept_
params0 = logistic.create_params_from_array(w.T,b)


########################################################
# This part is to search for some counterfactuals to test
########################################################
"""
adult_train['predictions'] = log_reg.predict(X_train)
# Find rows meeting the criteria in the training dataset
filtered_rows = adult_train[
    (adult_train['race_White'] == 1) & 
    (adult_train['gender_Male'] == 1) & 
    (adult_train['native-country_North-America'] == 1) & 
    (adult_train['hours-per-week'] >= 0.5) & 
    (adult_train['hours-per-week'] <= 0.7) & 
    (adult_train['predictions'] == 1)
]

# Map back to the original dataset indices
original_indices = filtered_rows.index.tolist()
"""
########################################################
########################################################

params_state = key, params0
F = logistic.F_function(X_train, y_train, linear, betaF)
hypsF = F, jax.grad(F), etaF
_, traj_params = langevin.MALA_chain(params_state, hypsF, N_params)

a = jax.vmap(F)(traj_params)/betaF
print(f"parameter trajectory quality max: {jnp.max(a)}, min:{jnp.min(a)}, mean:{jnp.mean(a)}, std:{jnp.std(a)}")

betaG= 1000.
etaG = 0.01/betaG
etaG = utils.sqrt_decay(etaG)
N_x = 5000

x_b = X_train[499]  
xs = X_train[499]  
state_x = key, x_b

#neg_predictor = logistic.negation_logistic_estimator(params0, linear)


G = logistic.G_function(traj_params, linear, logistic.constant_estimator(1.), logistic.cross_entropy, partial(logistic.l2_reg, C = 0.01, x0 = xs), betaG)
#G = logistic.G_function(traj_params, linear, neg_predictor, logistic.cross_entropy, partial(logistic.l2_reg, C = 0., x0 = x0), betaG)

#lower_bound = jnp.zeros_like(xs)
#upper_bound = jnp.ones_like(xs)

indices = jnp.array([0, 1, 2])

# New values for these indices
new_values_lower = jnp.array([0.2, 0.3, 0.4])
new_values_upper = jnp.array([0.32,0.48,0.5])


lower_bound = jnp.zeros_like(xs).at[indices].set(new_values_lower)
upper_bound = jnp.ones_like(xs).at[indices].set(new_values_upper)


hypsG = G, jax.grad(G), etaG, lower_bound, upper_bound 

_, traj_x = langevin.MALA_chain(state_x, hypsG, N_x)

a = jax.vmap(G)(traj_x)/betaG
print(f"x trajectory quality max: {jnp.max(a)}, min:{jnp.min(a)}, mean:{jnp.mean(a)}, std:{jnp.std(a)}")

ys = log_reg.predict(traj_x)
data_path = jnp.column_stack([traj_x, ys])
data_path_df = pd.DataFrame(data_path, columns= adult.columns)
inverted = create_datasets.invert_adult(data_path_df, preprocessor)


# summarize_results.visualize_samples(inverted[-500:])

# Define features
numerical_features = ["age", "educational-num", "hours-per-week"]
categorical_features = ["race", "gender", "native-country", "workclass", "occupation", "relationship"]

# Plot
factual = df.iloc[7209].to_dict()

# Get the dictionaries
region_dict, workclass_dict, relationship_dict = create_datasets.get_dict()

# Update 'factual' using the dictionaries
if "native-country" in factual:
    factual["native-country"] = region_dict.get(factual["native-country"], factual["native-country"])
if "workclass" in factual:
    factual["workclass"] = workclass_dict.get(factual["workclass"], factual["workclass"])
if "relationship" in factual:
    factual["relationship"] = relationship_dict.get(factual["relationship"], factual["relationship"])


summarize_results.summary_plots(factual , inverted[-500:],  
                               numerical_features=numerical_features, 
                               categorical_features=categorical_features)


