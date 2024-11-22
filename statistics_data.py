#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:46:20 2024

@author: u1573378
"""

import numpy as np
import pandas as pd


def decode_synthetic_instance(x_i):
    """
    Decodes the encoded columns in a synthetic data instance x_i.

    Args:
        x_i (numpy.ndarray): A single synthetic data instance as a numpy array.

    Returns:
        tuple: Decoded values for workclass, race, and gender.
    """
    # Indices for encoded columns
    workclass_indices = [7, 8, 9]  # workclass_government, workclass_others, workclass_private
    race_index = 10  # race_white
    gender_index = 11  # gender_Male

    # Decode workclass using softmax
    workclass_logits = x_i[0, workclass_indices]
    workclass_probabilities = np.exp(workclass_logits) / np.sum(np.exp(workclass_logits))
    workclass_categories = ['government', 'others', 'private']
    workclass_decoded = workclass_categories[np.argmax(workclass_probabilities)]

    # Decode race and gender
    race_decoded = 'white' if x_i[0, race_index] > 0.5 else 'others'
    gender_decoded = 'Male' if x_i[0, gender_index] > 0.5 else 'Female'

    return workclass_decoded, race_decoded, gender_decoded


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

def compare_categorical_changes(last_xs, x_0, decode_synthetic_instance):
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
    decoded_last_xs = [decode_synthetic_instance(x.reshape(1, -1)) for x in last_xs]
    decoded_x_0 = decode_synthetic_instance(x_0.reshape(1, -1))
    
    # Convert to DataFrames for comparison
    df_last_xs = pd.DataFrame(decoded_last_xs, columns=["workclass", "race", "gender"])
    df_x_0 = pd.DataFrame([decoded_x_0] * len(last_xs), columns=["workclass", "race", "gender"])
    
    # Compare and count flips
    flip_summary = (df_last_xs != df_x_0).apply(pd.Series.value_counts).fillna(0).astype(int)
    flip_summary.columns = ["workclass_flips", "race_flips", "gender_flips"]
    
    return flip_summary





def decode_categorical_features(points, decode_synthetic_instance):
    decoded_features = []
    for point in points:
        decoded = decode_synthetic_instance(point.reshape(1, -1))  # Decode each point
        decoded_features.append(decoded)
    return pd.DataFrame(decoded_features, columns=["workclass", "race", "gender"])

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