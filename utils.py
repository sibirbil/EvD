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


###########################
## DATA PREPRATION UTILS ##
###########################

import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any

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
