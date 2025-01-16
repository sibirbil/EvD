
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import create_datasets

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

import nets, train, optax, utils

import logistic, langevin
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def G_contrast_function(lin_reg, svr_model, beta):
    w = jnp.array(lin_reg.coef_)  # Shape: (n_features,)
    b = jnp.array(lin_reg.intercept_)  # Shape: ()
    
    # Extract support vectors, dual coefficients, and intercept
    support_vectors = svr_model.support_vectors_
    dual_coefs = svr_model.dual_coef_.flatten()  
    b_svr = svr_model.intercept_[0]             

    # Compute the weight vector (w)
    w_svr = np.dot(dual_coefs, support_vectors)
    
    # just to check
    #w_svr2 = svr_model.coef_.flatten()
    #b_svr2 = svr_model.intercept_

    def G(x):
        # Linear regression prediction
        y1 = jnp.dot(w, x) + b  # Linear regression output

        # SVR prediction
        y2 = jnp.dot(w_svr, x) + b_svr

        # Compute contrastive loss
        diff = (y1 - y2)*1
        return beta * jnp.exp(-jnp.linalg.norm(diff) ** 2)
    
    return G

def G_similar_function(lin_reg, svr_model, beta):
    w = jnp.array(lin_reg.coef_)  # Shape: (n_features,)
    b = jnp.array(lin_reg.intercept_)  # Shape: ()
    
    # Extract support vectors, dual coefficients, and intercept
    support_vectors = svr_model.support_vectors_
    dual_coefs = svr_model.dual_coef_.flatten()  
    b_svr = svr_model.intercept_[0]             

    # Compute the weight vector (w)
    w_svr = np.dot(dual_coefs, support_vectors)

    def G(x):
        # Linear regression prediction
        y1 = jnp.dot(w, x) + b  # Linear regression output

        # SVR prediction
        y2 = jnp.dot(w_svr, x) + b_svr

        # Compute contrastive loss
        diff = y1 - y2
        return beta * jnp.linalg.norm(diff) ** 2
    
    return G


def print_max_min_values(data, labels, dataset_name):
    """Prints the maximum and minimum values for each feature """
    print(f"\nMax and Min values for {dataset_name}:")
    for i, label in enumerate(labels):
        max_val = np.max(data[:, i])
        min_val = np.min(data[:, i])
        print(f"Feature '{label}': Max = {max_val:.1f}, Min = {min_val:.1f}")



##################
### Visualisations
##################


import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def scatter_plot_with_reference(x, y, x_label, y_label, title, color="blue", alpha=0.6):
    """Plots a scatter plot with a reference y=x line."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=alpha, color=color)
    
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Agreement")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def boxplot_comparison(data, labels, title, y_label):
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.6), medianprops=dict(color="black"))
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Models')
    plt.tight_layout()
    plt.show()

def side_by_side_boxplots(data1, data2, labels, title, x_label, y_label):
    """Creates side-by-side boxplots for comparing two datasets"""
    
    data1 = np.array(data1)
    data2 = np.array(data2)

    n_features = data1.shape[1]  # Number of features

    fig, ax = plt.subplots(figsize=(14, 6))
    positions = np.arange(n_features)  # Base positions for features
    width = 0.35  # Width of each boxplot group

    # Plot boxplots for the first dataset
    ax.boxplot(
        [data1[:, i] for i in range(n_features)],  # Extract feature columns
        positions=positions - width / 2,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="blue", alpha=0.6),
        medianprops=dict(color="black"),
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='blue', markeredgecolor='blue')
    )

    # Plot boxplots for the second dataset
    ax.boxplot(
        [data2[:, i] for i in range(n_features)],  # Extract feature columns
        positions=positions + width / 2,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="green", alpha=0.6),
        medianprops=dict(color="black"),
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='green', markeredgecolor='green')
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    legend_handles = [
        Patch(facecolor="blue", edgecolor="black", label="Synthetic Data"),
        Patch(facecolor="green", edgecolor="black", label="Original Test Data")
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.show()



def feature_comparison_boxplots(similar_data, different_data, feature_names, title):
    """Creates feature-wise boxplots comparing similar and different prediction datasets."""
    n_features = similar_data.shape[1]

    plt.figure(figsize=(16, 6))

    colors = ["blue", "green"]

    for i in range(n_features):
        similar_values = similar_data[:, i]
        different_values = different_data[:, i]

        plt.boxplot(
            similar_values,
            positions=[i * 2],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[0])
        )

        plt.boxplot(
            different_values,
            positions=[i * 2 + 1],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[1])
        )

    plt.xticks([i * 2 + 0.5 for i in range(n_features)], feature_names, rotation=45, ha="right")
    plt.ylabel("Values")
    plt.title(title)
    plt.legend([
        Patch(facecolor="blue", label="Similar"),
        Patch(facecolor="green", label="Different")
    ], loc="upper right")

    plt.tight_layout()
    plt.show()

def compare_datasets(original, inverted, categorical_cols=None, numerical_cols=None):
    """
    Compare the original dataset with the inverted dataset.

    Parameters:
    - original (pd.DataFrame): The original dataset.
    - inverted (pd.DataFrame): The inverted dataset.
    - categorical_cols (list): List of categorical column names.
    - numerical_cols (list): List of numerical column names.
    """
    # Replace inf values with NaN
    original = original.replace([np.inf, -np.inf], np.nan)
    inverted = inverted.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN values to avoid plotting issues
    original = original.dropna()
    inverted = inverted.dropna()

    # Determine columns if not provided
    if categorical_cols is None:
        categorical_cols = original.select_dtypes(include=['object', 'category']).columns.tolist()
    if numerical_cols is None:
        numerical_cols = original.select_dtypes(include=['number']).columns.tolist()
    
    # Compare numerical features
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(original[col], label='Original', fill=True, color='blue')
        sns.kdeplot(inverted[col], label='Synthetic', fill=True, color='orange')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    # Compare categorical features
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        original_counts = original[col].value_counts(normalize=True)
        inverted_counts = inverted[col].value_counts(normalize=True)
        comparison_df = pd.DataFrame({'Original': original_counts, 'Synthetic': inverted_counts})
        comparison_df.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
        plt.title(f'Comparison of {col}')
        plt.ylabel('Proportion')
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

