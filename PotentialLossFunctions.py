import numpy as np
import utils
import jax
import jax.numpy as jnp
import langevin
import jax.random as random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import create_datasets
from sklearn.metrics import confusion_matrix
from statistics_data import( 
    decode_categorical_features, compute_distances,
    compute_feature_changes, decode_synthetic_instance,
    compare_synthetic_instances, compare_categorical_changes
    )
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler


"""
from sklearn.datasets import make_classification
# Generate a synthetic dataset for a logistic regression 
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=1, 
                           n_classes=2, random_state=42)




#### GIVE ME SOME CREDIT DATASET

allnumeric = 1

df, X, y, column_names = create_datasets.get_gmsc()

#X = X.to_numpy().astype(float)
y = y.to_numpy()

# # Split the dataset
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=10)
# X = train_df.drop(columns=['SeriousDlqin2yrs'])
# y = train_df['SeriousDlqin2yrs']
# X_test = test_df.drop(columns=['SeriousDlqin2yrs'])
# y_test = test_df['SeriousDlqin2yrs']

#####


##### ADULT DATASET

allnumeric = 0  # the dataset is all numeric, no need to have decoding

df, encoded_cols = create_datasets.get_adult()

# Split the dataset
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=10)
#X = train_df.drop(columns=['income'])
#y = train_df['income']

X = df.drop(columns=['income'])
y = df['income']
column_names = X.columns

X = X.to_numpy().astype(float)
y = y.to_numpy()

##########


#### FICO

allnumeric = 1

df, X, y, column_names = create_datasets.get_fico()

#X = X.to_numpy().astype(float)
y = y.to_numpy()

# # Split the dataset
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=10)
# X = train_df.drop(columns=[RiskPerformance'])
# y = train_df['RiskPerformance']
# X_test = test_df.drop(columns=['RiskPerformance'])
# y_test = test_df['RiskPerformance']

#####

"""

######################
# Istedigin datasetini yukaridan kopyalayip asagida deneyebilirsin
######################


##### ADULT DATASET

allnumeric = 0  # the dataset is all numeric, no need to have decoding

df, original_df, encoded_cols = create_datasets.get_adult()

# # Define the categories you want to drop
# categories_to_drop = ['occupation', 'marital-status', 'relationship', 'native-country']  # Add more categories as needed

# # Collect all columns to drop
# columns_to_drop = []
# for category in categories_to_drop:
#     if category in encoded_cols:
#         columns_to_drop.extend(encoded_cols.pop(category).keys())  # Remove the category and collect column names

# # Drop the columns from the DataFrame
# df = df.drop(columns=columns_to_drop)


# # Update indices in encoded_cols to reflect the updated DataFrame
# encoded_cols = {
#     key: {col: df.columns.get_loc(col) for col in cols.keys() if col in df.columns}
#     for key, cols in encoded_cols.items()
#     }



# Split the dataset
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=10)
#X = train_df.drop(columns=['income'])
#y = train_df['income']

X = df.drop(columns=['income'])
y = df['income']
column_names = X.columns

X = X.to_numpy().astype(float)
y = y.to_numpy()

##########


# train a logistic regression model
log_reg = LogisticRegression(max_iter = 15000)
log_reg.fit(X, y)

# y values between 0 and 1
y_prob = log_reg.predict_proba(X)[:,1] 

# Training data
# retrieve the model's theta values with the intercept
theta_values = list(log_reg.coef_[0]) + [log_reg.intercept_[0]]  # Add intercept to coefficients
D = jnp.hstack([jnp.array(X), jnp.ones((X.shape[0], 1))])
logits = D @ jnp.array(theta_values)
y_prob_train = 1/(1 + jnp.exp(-logits))  

y_pred_train = np.zeros(len(y))
for i in range(len(y)):
    if y_prob_train[i] > 0.5:
        y_pred_train[i] = 1

confusion_matrix(y, y_pred_train) 

"""
# Test Data
X_test = X_test.to_numpy().astype(float)
y_test = y_test.to_numpy()

# Predict on the test set
y_pred_test = log_reg.predict(X_test)

# Compute the confusion matrix
result_test = confusion_matrix(y_test, y_pred_test)
"""


X = jnp.array(X)
y = jnp.array(y)


def F_function(
    X    : jax.Array, 
    y    : jax.Array,
    lambda_reg,
    reg_type="L2" 
    ):
    
    D = jnp.hstack([X, jnp.ones(X.shape[0]).reshape(-1,1)])
    
    """
    Logistic regression loss function:
    F(theta) = (1/N) * sum_i [log(1 + exp(x_i^T theta)) - y_i * x_i^T theta]
    """
    
    def F(theta):
        logits = D @ theta
        #term1 = jnp.log(1 + jnp.exp(logits))  # log(1 + exp(x_i^T theta))
        term1 = -jax.nn.log_sigmoid(-logits)
        term2 = y * logits
        base_loss = jnp.mean(term1 - term2)
        
        if reg_type == "L2":
            regularizer = lambda_reg * jnp.sum(theta**2)  # L2 regularization
        else: # reg_type == "L1":
            regularizer = lambda_reg * jnp.sum(jnp.abs(theta))  # L1 regularization
           
        return base_loss + regularizer
 

    return F, jax.grad(F)


# theta_0 = jnp.array([1.,1.,1.,1.,1.,1.])
len_vec = X.shape[1]+1
theta_0 = jnp.array(0.05*np.ones(len_vec))
step_size_gd = 0.0001  # Gradient descent step size
num_gd_steps = 10000  # Number of steps


# Gradient descent
F, gradF = F_function(X, y, lambda_reg=0.1, reg_type="L2")
theta = theta_0
# loss_values = []
# tolerance = 0.000001  

# for i in range(num_gd_steps):
#     loss = F(theta)
#     loss_values.append(loss)  # Store current loss
    
#     # Stopping condition
#     if i > 0 and abs(loss_values[-2] - loss) / loss_values[-2] < tolerance:
#         break
    
#     gradients = gradF(theta)
#     theta = theta - step_size_gd * gradients
    
#     # Additional stop condition for small gradients
#     if jnp.linalg.norm(gradients) < 1e-3:
#         break

theta_0 = jnp.array(np.array(theta_values)) #theta


key = random.PRNGKey(24)
key, subkey = random.split(key)


state_theta = (key, theta_0)   #(theta_0, key)
hypsF = (F, gradF, 0.001) #last entry is step size

(last_key, last), traj_theta = langevin.MALA_chain(state_theta, hypsF, 100000)

thetas = traj_theta[99900:]
thetas = theta_0

def G_function(
        thetas      :jax.Array,
        y_val       :jnp.float_,
        x_s         :jax.Array,
        lambda_T1   :jnp.float_,   # for regularization type1
        lambda_T2   :jnp.float_,   # for regularization type2
        loss_type   :int           # 1 for Loss 1 (close to 0.5), 2 for Loss 2 (counterfactual)
):
    
    def G(x):
        if loss_type == 1:
            # Loss 1
            d = jnp.hstack([x, 1])
            logits = thetas @ d
            #f_xtheta = 1 / (1 + jnp.exp(-logits))  
            f_xtheta = jax.nn.sigmoid(logits)
            base_loss = jnp.mean(jnp.square(f_xtheta - y_val))*50
            # lambda_T1 = 1
            return base_loss + lambda_T1*jnp.sum(x**2)
        
        elif loss_type == 2:
            # Loss 2
            squared_distance = jnp.sum((x_s - x)**2)
            
            d = jnp.hstack([x, 1])
            logits = thetas @ d
            #term1 = jnp.log(1 + jnp.exp(logits))  # log(1 + exp(x_i^T theta))
            term1 = -jax.nn.log_sigmoid(-logits)
            term2 = y_val * logits 
            difference = jnp.mean(term1 - term2)
            
            return squared_distance + lambda_T2 * difference
        
        else:
            raise ValueError("Invalid loss_type.")
    
    return G, jax.grad(G)


#x_0 = jnp.array([0.5, 0.5, 0.5, 2., -1.])  # initial point #jnp.array(np.zeros(31))

# These points are picked for the Adult dataset
x_0 = jnp.array(X[10])   #private, white, male, 1
#x_0 = jnp.array(X[45])  #private, white, female, 1
#x_0 = jnp.array(X[319]) #others, male, 1
#x_0 = jnp.array(X[346]) #others, female, 1
#x_0 = jnp.array(X[1])   #private, white, male, 0
#x_0 = jnp.array(X[4])   #private, white, female, 0
#x_0 = jnp.array(X[6])   #private, others, male, 0
#x_0 = jnp.array(X[48])   #private, others, female, 0

# These points are picked for the GMSC dataset
#x_0 = jnp.array(X[9])  # y_prob = 0.26, this point is for loss type 1
#x_0 = jnp.array(X[3])  # y_prob = 0.15, this point is for loss type 1

# These points are picked for the FICO dataset
#x_0 = jnp.array(X[1])  # y_prob = 0.93, this point is for loss type 1
#x_0 = jnp.array(X[7])  # y_prob = 0.34 this point is for loss type 2


# Set the loss_type to select the loss function
# Set to 1 for Loss function 1 (close to the boudnary 0.5)
# or 2 for Loss function 2 (counterfactual)
loss_type = 1  

if loss_type == 1:
    # Parameters for Loss function 1
    x_s = []  # Not needed
    G, gradG = G_function(thetas, 0.5, x_s, lambda_T1=1, lambda_T2=1, loss_type=1)

elif loss_type == 2:
    # Parameters for Loss function 2
    # x_s = jnp.array([0.1, 0.3, -1.2, 0.4, 0.1])  # Specific point for Loss function 2
    x_s = jnp.array(X[21])   # for Adult dataset (black, private, female)
    #x_s = jnp.array(X[10])   # for FICO dataset, y_prob = 0.27 
    #x_s = jnp.array(X[0])   # for GMSC dataset y_prob = 0.15  
    G, gradG = G_function(thetas, 1, x_s, lambda_T1=1, lambda_T2=10, loss_type=2)

else:
    raise ValueError("Invalid loss_type.")


state_x = (subkey, x_0)
hypsG = G, gradG, utils.sqrt_decay(0.01)

_, traj_x = langevin.MALA_chain(state_x, hypsG, 10000)

last_xs = traj_x[-100:]

# check only one point
sample_point = np.array(last_xs[-1:])

# apply inverse transform
# numeric_indices = [column_names.index(col) for col in numeric_columns]
# sample_point[:, numeric_indices] = scaler.inverse_transform(sample_point[:, numeric_indices])


if allnumeric == 0:
    decoded_instance = decode_synthetic_instance(sample_point, encoded_cols)

x_sample = sample_point

if loss_type == 1:
    comparison_df = compare_synthetic_instances(column_names, [], x_sample, x_0)
else:
    comparison_df = compare_synthetic_instances(column_names, x_s, x_sample, x_0)


if allnumeric == 0:
    # Decoding categorical features from the last 100 points
    decoded_categorical = decode_categorical_features(last_xs, decode_synthetic_instance, encoded_cols)

    # Print summary of decoded categorical features
    categorical_summary = decoded_categorical.apply(pd.Series.value_counts).fillna(0)
    print("Categorical Decoding Summary:")
    print(categorical_summary)


# Check predictions
y_prob_check = log_reg.predict_proba(last_xs)[:, 1] 
predicted_y = (y_prob_check > 0.5).astype(int)

if allnumeric ==0:
    # Add predicted
    decoded_categorical["predicted_y"] = predicted_y

    # Compare categorical changes
    categorical_flip = compare_categorical_changes(last_xs, x_0, decode_synthetic_instance, encoded_cols)

    print("Categorical Changes Summary:")
    print(categorical_flip)


# Compute Euclidean distances between each point in last_xs and x_0
distances = compute_distances(last_xs, x_0)

if loss_type == 2:
    distances_xs = compute_distances(last_xs, x_s)
    print("Distance from the target point (x_s):")
    for stat, value in distances_xs.items():
        print(f"{stat}: {value:.4f}")
    

# Print the statistics
print("Distance from Initial Point (x_0):")
for stat, value in distances.items():
    print(f"{stat}: {value:.4f}")
    
feature_changes = compute_feature_changes(last_xs, x_0, column_names)

# Print the statistics
print("Feature Changes Statistics:")
print(feature_changes)

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(y_prob_check, bins=20, edgecolor='black', color='skyblue', alpha=0.7)
plt.axvline(0.5, color='red', linestyle='dashed', linewidth=1, label='0.5')

plt.xlabel('Prediction Probabilities')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Probabilities Around 0.5')
plt.legend()
plt.show()


