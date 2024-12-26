import jax.numpy as jnp
from numpy import ndarray


######################
## SCHEDULERS ET AL ##
######################

def as_scheduler(value):
    """
    Turns scalar into constant step-size function
    """
    if callable(value):
        return value
    return lambda step: value


def power_decay(
    init_lr : jnp.float_,         # the starting learning rate 
    alpha   : jnp.float_,         # decay rate exponent
    offset  : jnp.float_  = 1.,   # in case step count starts from 0
    rate    : int | float = 100   # how many steps  
    ):
    """
    Returns a scheduler which decays by 1/(step/rate + 1)^alpha.
    The rate determines how many steps it takes to 
    """
    def schedule(step: int)-> float:
        return init_lr/ ((step/rate + offset)**alpha)
    
    return schedule

def sqrt_decay(init_lr):
    return power_decay(init_lr, 1/2)

def harmonic_decay(init_lr):
    return power_decay(init_lr, 1)


##############################
## MAKE PARAMS a TRAJECTORY ##
##############################

def make_traj(params):
    """
    add a vacuous leading dimension
    so that it can be fed into F funciton.
    """
    f = lambda x : jnp.expand_dims(x, 0)
    return jax.tree.map(f, params)


###########################
## DATA PREPRATION UTILS ##
###########################

import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any
from jax import random as random
import jax

def _make_probability(weights):
    """
    Alternative to softmax, when the inputs aren't log-probabilities.
    Clips the input to [0,1] and normalizes by their sum. If extra_dim
    is True, then assumes there is an extra catch-all category.
    """
    weights = jnp.maximum(0., jnp.minimum(1., weights)) #clip to [0,1]
    total = sum(weights) + 1e-4
    return weights/total

def _re_one_hotify_one_row(key, weights):
    """
    Return a one-hot vector of the same dimension as weights with 
    probability as given by the probalities input.
    """
    probs = _make_probability(weights)
    a =  random.choice(key, len(weights), probs)
    return jax.nn.one_hot(a, num_classes = len(weights))

def re_one_hotify_probabilistic(key, X :jax.Array, idx:slice):
    coi = X[:, idx] #columns of interest
    one_hots = jax.vmap(_re_one_hotify_one_row)(coi, random.split(key, X.shape[0]))
    return X.at[:, idx].set(one_hots)

def re_one_hotify_argmax(X: jax.Array, idx : slice):
    coi = X[:, idx] # columns of interest
    one_hots = jax.vmap(lambda v: (v>=jnp.max(v)).astype(float))(coi)
    return X.at[:, idx].set(one_hots)

def get_indices_with_prefix(df : pd.DataFrame, prefix: str):
    """
    Returns a slice of indices of columns that start with the given prefix. 
    All such columns are assumed to be contiguious.
    """
    prefix_cols = df.columns[df.columns.str.startswith(prefix)]
    idx = [df.columns.get_loc(col) for col in prefix_cols]
    return slice(idx[0], idx[-1]+1)

def _modify_to_series(X:pd.Series | pd.DataFrame):
        if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            raise ValueError("WeightedImputer can only fit on a single column.")

        # If X is a DataFrame with one column, convert it to a Series
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return X

class WeightedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X  : pd.DataFrame | pd.Series, _ : Any | None =None):
        X = _modify_to_series(X)
        self.category_frequencies_ = X.value_counts(normalize=True)
        return self

    def transform(self, X : pd.Series | pd.DataFrame):
        X = _modify_to_series(X)
        # Ensure that we are dealing with categorical data (object dtype)
        if not np.issubdtype(X.dtype, np.object_):
            raise ValueError("RandomImputer can only be applied to categorical data.")

        missing_indices = X.isna()
        
        # Randomly sample from the available categories according to their frequencies
        if missing_indices.any():
            imputed_values = np.random.choice(
                self.category_frequencies_.index,
                size=missing_indices.sum(),
                p=self.category_frequencies_.values
                )

        # Replace missing values with the sampled values
        X_copy = X.copy()
        X_copy.loc[missing_indices] = imputed_values
        return X_copy.to_frame()
    
    def get_feature_names_out(self, input_features = None):
        return input_features
        

def fn_from_dict(d : dict):
    
    f = lambda k : d.get(k,k)
    def F(X):
        return np.vectorize(f)(X)
    return F
