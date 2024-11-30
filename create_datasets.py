import jax
import jax.numpy as jnp
import jax.random as random

import numpy as np


##########################
####    IMAGE DATA    ####
##########################

# We load data from Huggingface, they are of Dict or Dataset type
# with 'color' channel added in if it doesn't exist. Values are
# standardized to be in [0,1]. Labels are integers (not one-hot).
# The column names are 'image' and 'label'


from datasets import load_dataset, Features, Array3D, Dataset


def _standardize_image_pixels(example):
    example["image"] = example['image']/255.0  
    return example

def _expand_channel_dim(example):
    example['image'] = jnp.expand_dims(example['image'], axis = -1)
    return example

def get_MNIST(
    split           : str,          # 'train' or 'test'
    color_channel   : bool = True   # whether to add the color channel (trivially)
    ) -> Dataset:
    """
    Returns a Dataset object with two columns, 'image', and 'label'.
    'image' is a float jax.Array of shape (N, 28, 28, 1) with values in [0,1]
    and 'label' column is an int jax.Array of shape (N,) with values 0,...,9
    where N = 60k or 10k depending on whether the 'train' or 'test' split is taken.
    """
    MNIST = load_dataset("mnist", split = split)
    MNIST.set_format('numpy')
    MNIST = MNIST.map(_standardize_image_pixels)

    if color_channel:
        MNIST = MNIST.map(_expand_channel_dim)
        features = Features({**MNIST.features, 'image':Array3D(dtype = 'float32', shape = (28,28,1)) })

    nothing= lambda x : x
    MNIST = MNIST.map(nothing, features = features)
    
    MNIST.set_format("jax")
    return MNIST

def get_CIFAR(
    split   : str,          #'test' or 'train'
    ) -> Dataset:
    """
    Returns a Dataset objet with two columns, 'image' and 'label'.
    'image' column is a float jax.Array of shape (N, 32, 32, 3)
    and 'label' column is an int jax.Array of shape (N,) where
    N = 50k or 10k depending on whether 'train' or 'test' split is taken. 
    Labels are integers between 0 and 9 (inclusive).
    """
    CIFAR = load_dataset("cifar10", split=split)
    CIFAR = CIFAR.rename_column("img", "image")

    CIFAR.set_format("numpy")
    CIFAR = CIFAR.map(_standardize_image_pixels)
    
    # explicitly declaring the type of data at each row is critical for performance.
    features = Features({**CIFAR.features, 'image':Array3D(dtype = 'float32', shape=(32,32,3))})
    nothing = lambda x : x
    CIFAR = CIFAR.map(nothing, features = features)
    
    CIFAR.set_format("jax")
    return CIFAR


############################
####    TABULAR DATA    ####
############################
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
import pandas as pd
import os

import utils

file_paths = {
    'adult': 'data/adult.csv',
    'gmsc' : 'data/GiveMeSomeCredit.csv',
    'fico' : 'data/FICO_dataset.csv'
    }

def _check_file_paths(file_paths):
    for filename in file_paths.values():
        assert os.path.exists(filename), f"{filename} does not exist."

_check_file_paths(file_paths)

##
## ADULT
##

_region_dict = {
    'United-States': 'North-America',
    'Mexico': 'Latin-America',
    'Philippines': 'Indochina',
    'Germany': 'Western-Europe',
    'Puerto-Rico': 'Latin-America',
    'Canada': 'North-America',
    'El-Salvador': 'Latin-America',
    'India': 'Subcontinent',
    'Cuba': 'Latin-America',
    'England': 'Western-Europe',
    'China': 'East-Asia',
    'South': 'East-Asia',
    'Jamaica': 'Latin-America',
    'Italy': 'Western-Europe',
    'Dominican-Republic': 'Latin-America',
    'Japan': 'East-Asia',
    'Guatemala': 'Latin-America',
    'Poland': 'Eastern-Europe',
    'Vietnam': 'Indochina',
    'Columbia': 'Latin-America',
    'Haiti': 'Latin-America',
    'Portugal': 'Western-Europe',
    'Taiwan': 'East-Asia',
    'Iran': 'Subcontinent',
    'Greece': 'Eastern-Europe',
    'Nicaragua': 'Latin-America',
    'Peru': 'Latin-America',
    'Ecuador': 'Latin-America',
    'France': 'Western-Europe',
    'Ireland': 'Western-Europe',
    'Hong': 'East-Asia',
    'Thailand': 'Indochina',
    'Cambodia': 'Indochina',
    'Trinadad&Tobago': 'Latin-America',
    'Yugoslavia': 'Eastern-Europe',
    'Outlying-US(Guam-USVI-etc)': 'North-America',
    'Laos': 'Indochina',
    'Scotland': 'Western-Europe',
    'Honduras': 'Latin-America',
    'Hungary': 'Eastern-Europe',
    'Holand-Netherlands': 'Western-Europe',
}

_workclass_dict = {
    'Private': 'private',
    'Local-gov': 'government',
    'Federal-gov': 'government',
    'State-gov': 'government',
    'Self-emp-not-inc': 'others',
    'Self-emp-inc': 'others',
    'Without-pay': 'others', #too little in number, 
    'Never-worked': 'others' #too little in number 
    }

_education_order = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", 
    "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", 
    "Masters", "Prof-school", "Doctorate"
    ]

    
def get_adult():

    df = pd.read_csv(file_paths['adult'], na_values = '?')

    # different procedures will be applied to each column. Feature columns are:
    # 'age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 
    # 'fnlwgt', 'marital-status', 'relationship', 'race', 'workclass', 'occupation', 
    # 'native-country', 'education', 'gender',
    # The last column is 'income' which is considered to be the TARGET
    
    
    numerical_columns = [
        'age', 'educational-num', 'capital-gain',
        'capital-loss', 'hours-per-week', 'fnlwgt'
        ]
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
        ])


    categorical_columns = ['marital-status', 'relationship', 'race']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder())]            # 7 + 6 + 5 = 18 classes
        )
    

    workclass_consolidator = FunctionTransformer(utils.fn_from_dict(_workclass_dict))
    workclass_transformer = Pipeline(steps =[
        ('imputer', utils.WeightedImputer()),
        ('consolidate', workclass_consolidator),
        ('one_hot', OneHotEncoder())            # 3 classes after consolidation
    ])


    occupation_transformer = Pipeline(steps=[
        ('imputer', utils.WeightedImputer()),
        ('one_hot', OneHotEncoder())            #14 classes
    ])
    

    region_consolidator = FunctionTransformer(utils.fn_from_dict(_region_dict))
    country_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('consolidate', region_consolidator),
        ('one_hot', OneHotEncoder())            # 7 regions after consolidation
    ])
    
    
    education_encoder = OrdinalEncoder(categories=[_education_order])

    
    # putting it all together
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns),
        ('wrk', workclass_transformer, ['workclass']),
        ('occ', occupation_transformer, ['occupation']),
        ('nat', country_transformer, ['native-country']),
        ('edu', education_encoder, ['education']),
        ('gen', OrdinalEncoder(), ['gender']),
        ('inc', OrdinalEncoder(), ['income'])],
        verbose_feature_names_out=True
        )
    
    processed = preprocessor.fit_transform(df) # as a sparse matrix represeentation

    # unfortunately preprocessor.get_feature_names_out() doesn't work since not all of the 
    # transformers in the above pipeline (such as FunctionTransformer) has such a method defined. 
    # thus we have to get their names one by one, IN THE SAME ORDER, and concatenate to get column names.
    num = preprocessor.named_transformers_['num'].get_feature_names_out(input_features = numerical_columns)
    cat = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features = categorical_columns)
    wrk = preprocessor.named_transformers_['wrk'].named_steps['one_hot'].get_feature_names_out(input_features = ['workclass'])
    occ = preprocessor.named_transformers_['occ'].named_steps['one_hot'].get_feature_names_out(input_features = ['occupation'])
    nat = preprocessor.named_transformers_['nat'].named_steps['one_hot'].get_feature_names_out(input_features = ['native-country'])
    edu = preprocessor.named_transformers_['edu'].get_feature_names_out(input_features= ['education'])
    gen = preprocessor.named_transformers_['gen'].get_feature_names_out(input_features = ['gender'])
    inc = preprocessor.named_transformers_['inc'].get_feature_names_out(input_features = ['income'])

    # Rename the encoded gender column to 'gender_male'
    gen = ['gender_male' if name == 'gender' else name for name in gen]

    feature_names = np.concatenate([num, cat, wrk, occ, nat, edu, gen, inc])
    
    # Store the one-hot encoded columns and their indices
    one_hot_encoded_columns = {
        'workclass': {name: idx for idx, name in enumerate(feature_names) if name.startswith('workclass')},
        'race': {name: idx for idx, name in enumerate(feature_names) if name.startswith('race')},
        'marital-status': {name: idx for idx, name in enumerate(feature_names) if name.startswith('marital')},
        'relationship': {name: idx for idx, name in enumerate(feature_names) if name.startswith('relationship')},
        'occupation': {name: idx for idx, name in enumerate(feature_names) if name.startswith('occupation')},
        'native-country': {name: idx for idx, name in enumerate(feature_names) if name.startswith('native-country')},
        'gender': {name: idx for idx, name in enumerate(feature_names) if name.startswith('gender')}
    }
    
    
    processed_df = pd.DataFrame(processed.toarray(), columns = feature_names)

    return processed_df, one_hot_encoded_columns




#############################
#### MISC SYNTHETIC DATA ####
#############################

from sklearn.datasets import make_moons

def moons_dataset(key = random.PRNGKey(42), n_samples =1000, noise  =0.1):
    X, y = make_moons(n_samples= n_samples, noise=noise, random_state = key[0])
    X1 = jnp.array(X[y==1])
    X2 = jnp.array(X[y==0])
    t = 2*y-1
    return X1, X2, t

def create_mixture_of_gaussians(
        key         :random.PRNGKey,
        means       :list[jax.Array], 
        sqrt_covs   :list[jax.Array], 
        logits      :jax.Array, 
        num_samples :int
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


def default_mixture_of_gaussian():
    num_samples = 1000
    means1 = [jnp.array([0, 0]), jnp.array([0, 1]), jnp.array([1, 0])]
    means2 = [jnp.array([-1,-1]), jnp.array([1,-1]), jnp.array([-1,1])]
    sqrt_covs = [(1/10)*jnp.eye(2), (1/10)*jnp.eye(2), (1/10)*jnp.eye(2)]
    weights = jnp.array([0.3, 0.5, 0.2])

    # Generate samples
    key1 = random.PRNGKey(42)
    samples1 = create_mixture_of_gaussians(key1, means1, sqrt_covs, weights, num_samples)
    key2 = random.PRNGKey(641)
    samples2 = create_mixture_of_gaussians(key2, means2, sqrt_covs, weights, num_samples)
    return samples1, samples2
