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
# The column names are 'x' and 'y' for image and label.


from datasets import load_dataset, Features, Array3D, Dataset


def _standardize_image_pixels(example):
    example["x"] = example['x']/255.0  
    return example

def _expand_channel_dim(example):
    example['x'] = jnp.expand_dims(example['x'], axis = -1)
    return example

def get_MNIST(
    split           : str,          # 'train' or 'test'
    color_channel   : bool = True   # whether to add the color channel (trivially)
    ) -> Dataset:
    """
    Returns a Dataset object with two columns, 'x', and 'y' containing imge and label.
    'x' is a float jax.Array of shape (N, 28, 28, 1) with values in [0,1]
    and 'y' column is an int jax.Array of shape (N,) with values 0,...,9
    where N = 60k or 10k depending on whether the 'train' or 'test' split is taken.
    """
    MNIST = load_dataset("mnist", split = split)
    MNIST = MNIST.rename_columns({'image':'x', 'label':'y'})
    MNIST.set_format('numpy')
    MNIST = MNIST.map(_standardize_image_pixels)

    if color_channel:
        MNIST = MNIST.map(_expand_channel_dim)
        features = Features({**MNIST.features, 'x':Array3D(dtype = 'float32', shape = (28,28,1)) })

    nothing= lambda x : x
    MNIST = MNIST.map(nothing, features = features)
    
    MNIST.set_format("jax")
    return MNIST

def get_CIFAR(
    split   : str           #'test' or 'train'
    ) -> Dataset:
    """
    Returns a Dataset objet with two columns, 'x' and 'y'.
    'x' column is a float jax.Array of shape (N, 32, 32, 3)
    and 'y' column is an int jax.Array of shape (N,) where
    N = 50k or 10k depending on whether 'train' or 'test' split is taken. 
    Labels are integers between 0 and 9 (inclusive).
    """
    CIFAR = load_dataset("cifar10", split=split)
    CIFAR = CIFAR.rename_columns({"img":"x", "label":"y"})

    CIFAR.set_format("numpy")
    CIFAR = CIFAR.map(_standardize_image_pixels)
    
    # explicitly declaring the type of data at each row is critical for performance.
    features = Features({**CIFAR.features, 'x':Array3D(dtype = 'float32', shape=(32,32,3))})
    nothing = lambda x : x
    CIFAR = CIFAR.map(nothing, features = features)
    
    CIFAR.set_format("jax")
    return CIFAR


def get_CIFAR100(
    split   : str,                      # 'test' or 'train'
    labels  : str   = 'fine_label'      # 'fine_label' or 'course_label'
    ) -> Dataset:
    """
    Returns the CIFAR100 dataset
    """
    CIFAR = load_dataset('cifar100', split = split)
    columns = ['fine_label', 'coarse_label']
    columns.remove(labels)
    CIFAR = CIFAR.remove_columns(columns)   #only labels remains
    CIFAR = CIFAR.rename_columns({'img':'x', labels:'y'})
    CIFAR.set_format("numpy")
    CIFAR = CIFAR.map(_standardize_image_pixels)
    features = Features({**CIFAR.features, 'x': Array3D(dtype='float32', shape = (32,32,3))})
    nothing = lambda x: x
    CIFAR = CIFAR.map(nothing, features = features)
    CIFAR.set_format("jax")
    return CIFAR

def get_TinyImageNet(
    split   : str   # 'train' or 'test' (equivalently 'valid')
    ) -> Dataset:
    """
    Returns the Tiny Image Net dataset
    """
    if split == 'test':
        split = 'valid'
    TIN = load_dataset('Maysee/tiny-imagenet', split=split)
    TIN = TIN.rename_columns({'image':'x', 'label':'y'})
    TIN.set_format("numpy")
    TIN = TIN.map(_standardize_image_pixels)
    
    # explicitly declaring the type of data at each row is critical for performance.
    features = Features({**TIN.features, 'x':Array3D(dtype = 'float32', shape=(64,64,3))})
    nothing = lambda x : x
    TIN = TIN.map(nothing, features = features)
    
    TIN.set_format("jax")
    return TIN

############################
####    TABULAR DATA    ####
############################
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
import pandas as pd
import os

import utils

file_paths = {
    'adult': 'data/adult.csv',
    'gmsc' : 'data/GiveMeSomeCredit.csv',
    'fico' : 'data/FICO_dataset.csv',
    'housing' : 'data/housing.csv'    
    }

def _check_file_paths(file_paths):
    for filename in file_paths.values():
        assert os.path.exists(filename), f"{filename} does not exist."

_check_file_paths(file_paths)

##################
## ADULT
##################

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

_relationship_dict = {
    'Husband': 'married',
    'Not-in-family': 'no-family',
    'Other-relative': 'other-relative',
    'Own-child': 'Own-child',
    'Unmarried': 'unmarried',
    'Wife': 'married'
    }
    
def get_dict():
    """
    Returns the dictionaries for region, workclass, and relationship mappings.
    
    Returns:
        tuple: (_region_dict, _workclass_dict, _relationship_dict)
    """
    return _region_dict, _workclass_dict, _relationship_dict

def get_adult():

    df = pd.read_csv(file_paths['adult'], na_values = '?')

    # different procedures will be applied to each column. Feature columns are:
    # 'age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 
    # 'fnlwgt', 'marital-status', 'relationship', 'race', 'workclass', 'occupation', 
    # 'native-country', 'education', 'gender',
    # The last column is 'income' which is considered to be the TARGET
    
    # first we apply the non-invertible transforms 
    # such as imputing or consolidating categories. 
    
    workclass_consolidator = FunctionTransformer(utils.fn_from_dict(_workclass_dict))
    workclass_transformer = Pipeline(steps =[
        ('imputer', utils.WeightedImputer()),
        ('consolidate', workclass_consolidator),
    ])
    
    relationship_consolidator = FunctionTransformer(utils.fn_from_dict(_relationship_dict))
    relationship_transformer = Pipeline(steps =[
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('consolidate', relationship_consolidator),
    ])
    
    
    occupation_transformer = Pipeline(steps=[
        ('imputer', utils.WeightedImputer()),
    ])
    
    region_consolidator = FunctionTransformer(utils.fn_from_dict(_region_dict))
    country_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('consolidate', region_consolidator),
    ])

    #irreversible proprocessing
    irrev = ColumnTransformer(transformers=[
        ('wrk', workclass_transformer, ['workclass']),
        ('rlt', relationship_transformer, ['relationship']),
        ('occ', occupation_transformer, ['occupation']),
        ('nat', country_transformer, ['native-country'])
        ],
        remainder='passthrough',
        verbose_feature_names_out=True
        )

    # imputed and consolidated dataframe
    imp_and_cons = irrev.fit_transform(df)
    transformed_cols = ['workclass', 'relationship', 'occupation', 'native-country']
    untouched_cols = df.columns.drop(transformed_cols)
    reordered_cols = np.concatenate([transformed_cols, untouched_cols])
    imp_and_cons_df = pd.DataFrame(imp_and_cons, columns = reordered_cols)


    numerical_columns = [
        'age', 'educational-num', 
        #'capital-gain', 'capital-loss', 
        'hours-per-week', 
        #'fnlwgt'
        ]
    categorical_columns = [
        'race', 'gender', 'native-country', 'workclass', 
        'occupation', 'marital-status', 'relationship'
        ]
    
    numerical_scaler = MinMaxScaler()
    numerical_transformer = Pipeline(steps=[
        ('scaler', numerical_scaler)
        ]) 
    categorical_transformer = Pipeline(steps=[
        ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))   # 5 + 2 + 7 + 3 + 14 + 7 + 6 = 44 classes
    ])
    
    
    # putting it all together
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns),
        ('inc', OrdinalEncoder(), ['income'])],
        verbose_feature_names_out=False
        )
    
    processed = preprocessor.fit_transform(imp_and_cons_df) # as a sparse matrix represeentation

    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed, columns = feature_names)

    return processed_df, preprocessor, df


# def adult_one_hotify(
#     adult       : pd.DataFrame,      # DataFrame with expanded columns.
#     method      : str,               # 'argmax' or 'probabilistic'
#     key         : random.PRNGKey = None # needed for probabilistic method
#     ):
#     """
#     Returns a dataframe which returns the relevant columns as hot according to given method.
#     """
#     X = jnp.array(adult.to_numpy())
#     prefix_list = ['race_', 'gender_', 'native-country_', 'workclass_', 
#                    'occupation_', 'marital-status_', 'relationship_']
#     idxs = list(map(lambda p : utils.get_indices_with_prefix(adult, p), prefix_list))
#     if method == 'argmax':
#         for idx in idxs:
#             X = utils.re_one_hotify_argmax(X, idx)
#     if method == 'probabilistic':
#         for idx in idxs:
#             X = utils.re_one_hotify_probabilistic(key, X, idx)
#     return pd.DataFrame(X, columns = adult.columns)


def invert_adult(processed_df, preprocessor: ColumnTransformer):
    num_trans = preprocessor.named_transformers_['num']
    num_cols = num_trans.feature_names_in_

    cat_trans = preprocessor.named_transformers_['cat']
    cat_cols = cat_trans.get_feature_names_out()
    
    inv_cat = cat_trans.inverse_transform(processed_df[cat_cols])
    inv_num = num_trans.inverse_transform(processed_df[num_cols])

    num_df = pd.DataFrame(inv_num, columns = num_cols)
    cat_df = pd.DataFrame(inv_cat, columns = cat_trans.feature_names_in_)
    inc_df = pd.DataFrame(processed_df['income'])

    return pd.concat([num_df, cat_df, inc_df], axis=1)
    
##################
## GMSC
##################

def get_gmsc():

    df = pd.read_csv(file_paths['gmsc'])
    iqr_multiplier = 2  # for outliers

    # Impute missing values using the median for all input columns
    imputer = SimpleImputer(strategy='median')
    input_cols = df.columns  # All feature columns
    df[input_cols] = imputer.fit_transform(df[input_cols])


    # Remove outliers for relevant columns
    df = remove_outliers(df, 'DebtRatio', iqr_multiplier)
    df = remove_outliers(df, 'MonthlyIncome', iqr_multiplier)

    # Split the dataset into features (X) and target (y)
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return df, X, y, scaler


def remove_outliers(df, column_name, iqr_multiplier = 2):
    # TODO check this iqr multiplier default value if it is too stringent
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    df = df[(df[column_name] >= lower_bound) & \
            (df[column_name] <= upper_bound)]
    return df


##################
## FICO
##################

def get_fico():
    
    df = pd.read_csv(file_paths['fico'])
    # Replace -9, -8, and -7 with NaN
    df = df.applymap(lambda x: np.nan if x in [-9, -8, -7] else x)
    missing_percentage = df.isna().mean() * 100
    columns_to_drop = missing_percentage[missing_percentage > 25].index
    df_cleaned = df.drop(columns=columns_to_drop)
    df_cleaned = df_cleaned.drop_duplicates()
    
    X = df_cleaned.drop(columns = 'RiskPerformance')
    y = df_cleaned.RiskPerformance.replace(to_replace=['Bad', 'Good'], value=[1, 0])
    
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Initialize the scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    return df, X, y, scaler


##################
## Housing
##################

def get_housing():
    
    df = pd.read_csv(file_paths['housing'])
    
    # Replace categorical values 
    df['mainroad'] = df['mainroad'].replace({'yes': 1, 'no': 0})
    df['guestroom'] = df['guestroom'].replace({'yes': 1, 'no': 0})
    df['basement'] = df['basement'].replace({'yes': 1, 'no': 0})
    df['hotwaterheating'] = df['hotwaterheating'].replace({'yes': 1, 'no': 0})
    df['airconditioning'] = df['airconditioning'].replace({'yes': 1, 'no': 0})
    df['prefarea'] = df['prefarea'].replace({'yes': 1, 'no': 0})
    df['furnishingstatus'] = df['furnishingstatus'].replace({
        'unfurnished': 0, 
        'semi-furnished': 1, 
        'furnished': 2
        })
    
    X = df.drop(columns = 'price')
    y = df.price
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    scaler_y = MinMaxScaler()
    y = np.reshape(y, (-1,1))
    y = scaler_y.fit_transform(y)
                               
    
    return df, X, y, scaler, scaler_y





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
