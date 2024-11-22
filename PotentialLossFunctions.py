import numpy as np
import jax
import jax.numpy as jnp
import langevin
import jax.random as random
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from statistics_data import( 
    decode_categorical_features, compute_distances,
    compute_feature_changes, decode_synthetic_instance,
    compare_synthetic_instances, compare_categorical_changes
    )

"""
# Generate a synthetic dataset for a logistic regression 
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=1, 
                           n_classes=2, random_state=42)

"""

file = "./adult.csv"
df = pd.read_csv(file)

# some preparation steps
df.workclass = df.workclass.replace("?", "Private")
df.occupation.replace(to_replace='?',value=np.nan,inplace=True)
df['occupation'] = df['occupation'].fillna(method='bfill') 
df['native-country'] = df['native-country'].replace("?", "United-States")

# Apply label encoding to the 'education' column
label_encoder = LabelEncoder()
df['education'] = label_encoder.fit_transform(df['education'])


# separate numeric and categorical columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.difference(['income'])
categorical_columns = df.select_dtypes(include=['object']).columns

# initialize standardScaler and scale numeric columns
# scaler = StandardScaler()
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Update the income column to numerical values
df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

df['workclass'] = df['workclass'].replace({
    'Private': 'private',
    'Local-gov': 'government',
    'Federal-gov': 'government',
    'State-gov': 'government',
    'Self-emp-not-inc': 'others',
    'Self-emp-inc': 'others',
    'Without-pay': 'others',
    'Never-worked': 'Never-worked'
})

# remove the categorical variables
columns_to_remove = ['marital-status', 'occupation', 'relationship', 'native-country']
df.drop(columns=columns_to_remove, inplace=True)

# Group the 'race' column into 'white' and 'others'
df['race'] = df['race'].apply(lambda x: 'white' if x.lower() == 'white' else 'others')

# Apply one-hot encoding to the remaining categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)


# normalize some numerical columns
df_encoded['fnlwgt'] = df_encoded['fnlwgt']/max(df_encoded['fnlwgt']) 
df_encoded['capital-gain'] = df_encoded['capital-gain']/max(df_encoded['capital-gain'])
df_encoded['capital-loss'] = df_encoded['capital-loss']/max(df_encoded['capital-loss'])
  
X = df_encoded.drop(columns=['income'])
column_names = X.columns.tolist()

X = X.to_numpy()
X = X.astype(float)
y = df_encoded['income'].to_numpy()

#X = df.iloc[:,0:-1].to_numpy()
#y = df["class"].to_numpy()

#scaler = StandardScaler()
#X = scaler.fit_transform(X)


# train a logistic regression model
log_reg = LogisticRegression(max_iter = 15000)
log_reg.fit(X, y)

# y values between 0 and 1
y_prob = log_reg.predict_proba(X)[:,1] 


# retrieve the model's theta values with the intercept
theta_values = list(log_reg.coef_[0]) + [log_reg.intercept_[0]]  # Add intercept to coefficients
D = jnp.hstack([jnp.array(X), jnp.ones((X.shape[0], 1))])
logits = D @ jnp.array(theta_values)
y_prob_test= 1/(1 + jnp.exp(-logits))  

y_pred_test = np.zeros(len(y))
for i in range(len(y)):
    if y_prob_test[i] > 0.5:
        y_pred_test[i] = 1
        

X = jnp.array(X)
y = jnp.array(y)


def F_function(
    X    : jax.Array, 
    y    : jax.Array
    ):
    
    D = jnp.hstack([X, jnp.ones(X.shape[0]).reshape(-1,1)])
    
    """
    Logistic regression loss function:
    F(theta) = (1/N) * sum_i [log(1 + exp(x_i^T theta)) - y_i * x_i^T theta]
    """
    
    def F(theta):
        logits = D @ theta
        term1 = jnp.log(1 + jnp.exp(logits))  # log(1 + exp(x_i^T theta))
        term2 = y * logits 
        return jnp.mean(term1 - term2)

    return F, jax.grad(F)


# theta_0 = jnp.array([1.,1.,1.,1.,1.,1.])
len_vec = X.shape[1]+1
theta_0 = jnp.array(0.05*np.ones(len_vec))
step_size_gd = 0.0001  # Gradient descent step size
num_gd_steps = 10000  # Number of steps


# Gradient descent
F, gradF = F_function(X, y)
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


key = random.PRNGKey(42)
key, subkey = random.split(key)


state_theta = (key, theta_0)   #(theta_0, key)
hypsF = (F, gradF, 0.0001) #last entry is step size

(last_key, last), traj_theta = langevin.MALA_chain(state_theta, hypsF, 100000)

thetas = traj_theta[99000:]

def G_function(
        thetas      :jax.Array,
        y_val       :jnp.float_,
        x_s         :jax.Array,
        lambda_     :jnp.float_,   # for regularization 
        loss_type   :int           # 1 for Loss 1 (close to 0.5), 2 for Loss 2 (counterfactual)
):
    
    def G(x):
        if loss_type == 1:
            # Loss 1
            d = jnp.hstack([x, 1])
            logits = thetas @ d
            f_xtheta = 1 / (1 + jnp.exp(-logits))  
            return jnp.mean(jnp.square(f_xtheta - y_val))*100
        
        elif loss_type == 2:
            # Loss 2
            squared_distance = jnp.sum((x_s - x)**2)
            
            d = jnp.hstack([x, 1])
            logits = thetas @ d
            term1 = jnp.log(1 + jnp.exp(logits))  # log(1 + exp(x_i^T theta))
            term2 = y_val * logits 
            difference = jnp.mean(term1 - term2)
            
            return squared_distance + lambda_ * difference
        
        else:
            raise ValueError("Invalid loss_type.")
    
    return G, jax.grad(G)


#x_0 = jnp.array([0.5, 0.5, 0.5, 2., -1.])  # initial point #jnp.array(np.zeros(31))
x_0 = jnp.array(X[10])   #private, white, male, 1
#x_0 = jnp.array(X[45])  #private, white, female, 1
#x_0 = jnp.array(X[319]) #others, male, 1
#x_0 = jnp.array(X[346]) #others, female, 1
#x_0 = jnp.array(X[1])   #private, white, male, 0
#x_0 = jnp.array(X[4])   #private, white, female, 0
#x_0 = jnp.array(X[6])   #private, others, male, 0
#x_0 = jnp.array(X[48])   #private, others, female, 0

# Set the loss_type to select the loss function
# Set to 1 for Loss function 1 (close to the boudnary 0.5)
# or 2 for Loss function 2 (counterfactual)
loss_type = 1  

if loss_type == 1:
    # Parameters for Loss function 1
    x_s = []  # Not needed
    G, gradG = G_function(thetas, 0.5, x_s, lambda_=1, loss_type=1)

elif loss_type == 2:
    # Parameters for Loss function 2
    # x_s = jnp.array([0.1, 0.3, -1.2, 0.4, 0.1])  # Specific point for Loss function 2
    x_s = jnp.array(X[21]) 
    G, gradG = G_function(thetas, 1, x_s, lambda_=10, loss_type=2)

else:
    raise ValueError("Invalid loss_type.")


state_x = (subkey, x_0)
hypsG = G, gradG, 0.0001

_, traj_x = langevin.MALA_chain(state_x, hypsG, 10000)

last_xs = traj_x[-100:]

# check only one point
sample_point = np.array(last_xs[-1:])

# apply inverse transform
# numeric_indices = [column_names.index(col) for col in numeric_columns]
# sample_point[:, numeric_indices] = scaler.inverse_transform(sample_point[:, numeric_indices])


decoded_instance = decode_synthetic_instance(sample_point)

x_sample = sample_point

if loss_type == 1:
    comparison_df = compare_synthetic_instances(column_names, [], x_sample, x_0)
else:
    comparison_df = compare_synthetic_instances(column_names, x_s, x_sample, x_0)


# Decoding categorical features from the last 100 points
decoded_categorical = decode_categorical_features(last_xs, decode_synthetic_instance)

# Print summary of decoded categorical features
categorical_summary = decoded_categorical.apply(pd.Series.value_counts).fillna(0)
print("Categorical Decoding Summary:")
print(categorical_summary)


# Check predictions
y_prob_check = log_reg.predict_proba(last_xs)[:, 1] 
predicted_y = (y_prob_check > 0.5).astype(int)

# Add predicted
decoded_categorical["predicted_y"] = predicted_y

# Compare categorical changes
categorical_flip = compare_categorical_changes(last_xs, x_0, decode_synthetic_instance)

print("Categorical Changes Summary:")
print(categorical_flip)


# Compute Euclidean distances between each point in last_xs and x_0
distances = compute_distances(last_xs, x_0)

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


