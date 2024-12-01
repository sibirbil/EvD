#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:46:20 2024

@author: u1573378
"""

import numpy as np
import pandas as pd


def decode_synthetic_instance(x_i, encoded_cols):
    
    """
    one_hot_encoded_cols: Dictionary mapping one-hot encoded columns
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    decoded_values = {}

    # Decode features with multiple categories
    for feature, columns in encoded_cols.items():
        # for binary feature
        if isinstance(columns, dict) and len(columns) == 1:
            col_name, col_idx = next(iter(columns.items())) 
            probability_male = sigmoid(x_i[:,col_idx])  # Apply sigmoid to get probability
            decoded_values[feature] = 'male' if probability_male > 0.5 else 'female'
        
        else:  # Handle multi-category features using softmax
        
            indices = list(columns.values())
            categories = list(columns.keys())
            
            logits = x_i[:,indices]
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            probabilities = exp_logits / np.sum(np.exp(logits))
            decoded_values[feature] = categories[np.argmax(probabilities)]

    return decoded_values


def compare_synthetic_instances(columns, x_s, x_sample, x_0):
    """
    Compares two synthetic data instances and maps their values to columns.

    Args:
        columns (list): List of column names corresponding to the features.
        x_s (numpy.ndarray): First synthetic data instance as a numpy array.
        x_sample (numpy.ndarray): Second synthetic data instance as a numpy array.

    Returns:
        pd.DataFrame: A DataFrame showing the comparison between the two instances.
    """
    if len(x_s) == 0:
        x_s_flat = np.full(len(columns), np.nan)  # Fill with NaNs if x_s is empty
    else:
        x_s_flat = np.array(x_s).flatten()  # Ensure x_s is a numpy array and flattened
    
    
    # Flatten arrays to ensure they are 1D
    x_sample_flat = x_sample.flatten()
    x_0_flat = x_0.flatten()
    
    # Create a comparison DataFrame
    comparison = pd.DataFrame({
        'Column': columns,
        'x_s': x_s_flat,
        'x_sample': x_sample_flat,
        'x_initial': x_0_flat
    })
    return comparison

def compare_categorical_changes(last_xs, x_0, decode_synthetic_instance, encoded_cols):
    """
    Compares decoded categorical variables between `last_xs` and `x_0`.

    Args:
        last_xs (numpy.ndarray): Array of numerical synthetic points.
        x_0 (numpy.ndarray): Initial numerical point.
        decode_func (function): Function to decode numerical values to categories.

    Returns:
        pd.DataFrame: Summary of categorical changes.
    """
    # Decode categorical variables
    decoded_last_xs = [decode_synthetic_instance(x.reshape(1, -1), encoded_cols) for x in last_xs]
    decoded_x_0 = decode_synthetic_instance(x_0.reshape(1, -1), encoded_cols)
    
    # Dynamically create columns based on one_hot_encoded_cols keys
    columns = list(encoded_cols.keys())
    
    # Convert to DataFrames for comparison
    df_last_xs = pd.DataFrame(decoded_last_xs)
    df_x_0 = pd.DataFrame([decoded_x_0] * len(last_xs))
    
    # Compare and count flips
    flip_summary = (df_last_xs != df_x_0).apply(pd.Series.value_counts).fillna(0).astype(int)
    flip_summary.columns = [f"{col}_flips" for col in columns]
    
    return flip_summary





def decode_categorical_features(points, decode_synthetic_instance, encoded_cols):
    decoded_features = []
    for point in points:
        decoded = decode_synthetic_instance(point.reshape(1, -1), encoded_cols)  # Decode each point
        decoded_features.append(decoded)
   
    return pd.DataFrame(decoded_features)


def compute_distances(last_xs, x_0):
    distances = np.linalg.norm(last_xs - x_0, axis=1)
    return {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "min": np.min(distances),
        "max": np.max(distances),
    }

def compute_feature_changes(last_xs, x_0, column_names):
    feature_changes = last_xs - x_0
    return pd.DataFrame({
        "mean_change": np.mean(feature_changes, axis=0),
        "std_change": np.std(feature_changes, axis=0),
        "min_change": np.min(feature_changes, axis=0),
        "max_change": np.max(feature_changes, axis=0),
    }, index=column_names)