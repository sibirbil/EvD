import jax
import jax.numpy as jnp
import jax.random as random
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(n_samples=1000, noise=.1, random_state = 42)
X1 = jnp.array(X[y==1])
X2 = jnp.array(X[y==0])
t = 2*y-1

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


from datasets import load_dataset, Features, Array2D, Array3D


def standardize_image_pixels(example):
    example["image"] = example['image']/255.0  
    return example

def one_hotify(example):
    example["label"] = jax.nn.one_hot(example['label'], 10)


def get_MNIST(
    split       : str,          # 'test' or 'train'
    one_hot     : bool = False  # labels as indices or one hot encoded
):
    MNIST = load_dataset("mnist", split = split)
    MNIST.set_format('numpy')
    MNIST = MNIST.map(standardize_image_pixels)
    if one_hot:
        MNIST = MNIST.map(one_hotify)

    features = Features({**MNIST.features, 'image':Array2D(dtype = 'float32', shape = (28,28) )})
    nothing= lambda x : x
    MNIST = MNIST.map(nothing, features = features)
    
    MNIST.set_format("jax")
    return MNIST

def get_CIFAR(
    split   : str,          #'test' or 'train'
    one_hot : bool = False  # labels as indices or one hot encoded
):
    CIFAR = load_dataset("cifar10", split=split)
    CIFAR = CIFAR.rename_column("img", "image")

    CIFAR.set_format("numpy")
    CIFAR = CIFAR.map(standardize_image_pixels)
    if one_hot:
        CIFAR = CIFAR.map(one_hotify)
    
    features = Features({**CIFAR.features, 'image':Array3D(dtype = 'float32', shape=(32,32,3))})
    nothing = lambda x : x
    CIFAR = CIFAR.map(nothing, features = features)
    
    CIFAR.set_format("jax")
    return CIFAR
