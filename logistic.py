import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

def create_mixture_of_gaussians(
        num_samples :int, 
        means       :list[jax.Array], 
        sqrt_covs   :list[jax.Array], 
        logits      :jax.Array, 
        key         :random.PRNGKey
        ) -> jax.Array:
    """
    Creates samples from a mixture of Gaussian distributions.

    Parameters:
    - num_samples: Total number of samples to generate.
    - means: list of means for each Gaussian component.
    - sqrt_cov: list of matrices A such that (A^T A)^{-1} is the covariance 
        matrix for each Gaussian component.
    - logits: unnormalized log-probabilities for each Gaussian component,
        so that softmax(logits) gives corresponding probabilities.
    - key: JAX random key.

    Returns:
    - samples: Generated samples from the mixture of Gaussians.
    """
    num_components = len(means)
    dim = means[0].shape[0]

    # Sample component indices based on weights
    component_indices = random.categorical(key, logits, shape=(num_samples,))
    
    # Generate samples from each Gaussian component
    samples = []
    for i in range(num_components):
        # Get the number of samples for this component
        num_samples_i = jnp.sum(component_indices == i)
        
        if num_samples_i > 0:
            # Sample from the Gaussian
            samples_i = random.normal(key, shape=(num_samples_i, dim))
            samples_i = samples_i @ sqrt_covs[i].transpose() + means[i]
            samples.append(samples_i)

    return jnp.concatenate(samples)

# Parameters

num_samples = 1000
means1 = [jnp.array([0, 0]), jnp.array([0, 1]), jnp.array([1, 0])]
means2 = [jnp.array([-1,-1]), jnp.array([1,-1]), jnp.array([-1,1])]
sqrt_covs = [(1/10)*jnp.eye(2), (1/10)*jnp.eye(2), (1/10)*jnp.eye(2)]
weights = jnp.array([0.3, 0.5, 0.2])

# Generate samples
key1 = random.PRNGKey(42)
samples1 = create_mixture_of_gaussians(num_samples, means1, sqrt_covs, weights, key1)
key2 = random.PRNGKey(641)
samples2 = create_mixture_of_gaussians(num_samples, means2, sqrt_covs, weights, key2)

# Plot the samples
def plot_samples(samples1, samples2 = None)->None:
    plt.figure(figsize=(8, 8))
    plt.scatter(samples1[:, 0], samples1[:, 1], color= 'blue', alpha=0.5)
    if samples2 is not None:
        plt.scatter(samples2[:,0], samples2[:, 1], color= 'red', alpha = 0.5)
    plt.title('Mixture of Gaussians')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.show()


def logistic_loss(
        xs      :jax.Array,         # data
        ts      :jax.Array,         # labels given as +1 -1
        thetas  :jax.Array,         # parameters, where last entry is the bias term
        )-> jnp.float_:
    x_appended = _append1(xs)
    loss = jnp.mean(jax.nn.softplus(-ts*(x_appended @ thetas.transpose()))) 
    return loss

def l1_reg(input: jax.Array):
    """
    l1 norm of the input, which can be a multidimensional array
    """
    return jnp.sum(jnp.absolute(input))

def l2_reg(input: jax.Array):
    """
    l2 norm of the input which can be a multidimensional array
    """
    input_flat = input.flatten()
    return jnp.inner(input_flat, input_flat)

def _append1(x:jax.Array)->jax.Array:
    """
    Returns a matrix with a column of appended 1's
    Input x is either a vector or a matrix
    """
    if len(x.shape)==1:
        x = x[jnp.newaxis,:]
    return jnp.column_stack([x, jnp.ones(len(x))])

def F_function(
        xs          : jax.Array,    # data points
        ts          : jax.Array,    # labels corresponding to data, +/- 1
        reg_type    : str,          # 'l1' or 'l2'
        reg_const   : jnp.float_,    # multiplier in front of regularizer
        beta        : jnp.float_    # inverse temperature
        ):
    reg_fn = l1_reg if reg_type == 'l1' else l2_reg 
    F = lambda theta: beta*(logistic_loss(xs,ts,theta) + reg_const*reg_fn(theta))
    return F, jax.grad(F)

def G_function(
        thetas      : jax.Array,
        t           : int,          # +/- 1
        reg_type    : str,
        reg_const   : jnp.float_,
        beta        : jnp.float_,
        ):
    reg_fn = l1_reg if reg_type == 'l1' else l2_reg
    G = lambda x : beta*(logistic_loss(x, jnp.array([t]), thetas) +  reg_const*reg_fn(x))
    return G, jax.grad(G)


from sklearn.datasets import make_moons


X, y = make_moons(n_samples=1000, noise=.1, random_state = 42)
X1 = jnp.array(X[y==1])
X2 = jnp.array(X[y==0])
t = 2*y-1

func = lambda theta: logistic_loss(X, t, theta, 0.1)
func_grad = jax.grad(func)


def plot_it(
        thetas  : jax.Array,
        X1      : jax.Array = None,
        X2      : jax.Array = None,
        newX    : jax.Array = None
        )-> None:
    # Define x values for plotting
    x_vals = jnp.linspace(-10, 10, 100)

    # Create the figure
    plt.figure(figsize=(10, 10))

    for row in thetas:
        a, b, c = row
        if b != 0:
            y_vals = -(a / b) * x_vals - (c / b)
            plt.plot(x_vals, y_vals, alpha=0.1)  # Adjust alpha for transparency
        else:
            # Vertical line at x = -c/a (when b == 0)
            plt.axvline(x=-c/a, color='r', alpha=0.1)
    
    if X1 is not None:
        plt.scatter(X1[:, 0], X1[:,1], c = 'blue', alpha = 0.5)
    if X2 is not None:
        plt.scatter(X2[:, 0], X2[:,1], c = 'red', alpha = 0.5)
    if newX is not None:
        plt.scatter(newX[:,0], newX[:, 1], c = 'green', alpha = 0.1)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundaries')
    plt.grid(True)
    plt.show()
