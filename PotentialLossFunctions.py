import numpy as np
import jax
import jax.numpy as jnp
import langevin
import jax.random as random
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate a synthetic dataset for a logistic regression 
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=1, 
                           n_classes=2, random_state=42)


# Train a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# y values between 0 and 1
y_prob = log_reg.predict_proba(X)[:,1] 


# Retrieve the model's theta values with the intercept
theta_values = list(log_reg.coef_[0]) + [log_reg.intercept_[0]]  # Add intercept to coefficients
D = jnp.hstack([X, jnp.ones(X.shape[0]).reshape(-1,1)])
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


theta_0 = jnp.array([1.,1.,1.,1.,1.,1.])
step_size_gd = 0.001  # Gradient descent step size
num_gd_steps = 1000  # Number of steps


# Gradient descent
F, gradF = F_function(X, y)
theta = theta_0
loss_values = []
tolerance = 0.0001  

for i in range(num_gd_steps):
    loss = F(theta)
    loss_values.append(loss)  # Store current loss
    
    # Stopping condition
    if i > 0 and abs(loss_values[-2] - loss) / loss_values[-2] < tolerance:
        break
    
    gradients = gradF(theta)
    theta = theta - step_size_gd * gradients
    
    # Additional stop condition for small gradients
    if jnp.linalg.norm(gradients) < 1e-3:
        break

theta_0 = theta


key = random.PRNGKey(42)
key, subkey = random.split(key)

state_theta = (theta_0, key)
hypsF = (F, gradF, 0.01) #last entry is step size

(last, last_key), traj_theta = langevin.MALA_chain(state_theta, hypsF, 100000)

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
            return jnp.mean(jnp.square(f_xtheta - y_val))
        
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


x_0 = jnp.array([0.5,0.5, 0.5, 2., -1.])  # initial point

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
    x_s = jnp.array([0.1, 0.3, -1.2, 0.4, 0.1])  # Specific point for Loss function 2
    G, gradG = G_function(thetas, 0.5, x_s, lambda_=1, loss_type=2)

else:
    raise ValueError("Invalid loss_type.")

state_x = (x_0, subkey)
hypsG = G, gradG, 0.00001

_, traj_x = langevin.MALA_chain(state_x, hypsG, 10000)

last_xs = traj_x[-1000:]


if loss_type == 1:
    # Check predictions
    y_prob_check = log_reg.predict_proba(last_xs)[:, 1] 
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob_check, bins=20, edgecolor='black', color='skyblue', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='dashed', linewidth=1, label='0.5')
    
    plt.xlabel('Prediction Probabilities')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities Around 0.5')
    plt.legend()
    plt.show()


