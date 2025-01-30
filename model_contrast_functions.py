
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
        diff = (y1 - y2)
        return beta * jnp.exp(-jnp.linalg.norm(diff) ** 2)
    
    return G


def print_max_min_values(data, labels, dataset_name):
    """Prints the maximum and minimum values for each feature """
    print(f"\nMax and Min values for {dataset_name}:")
    for i, label in enumerate(labels):
        max_val = np.max(data[:, i])
        min_val = np.min(data[:, i])
        print(f"Feature '{label}': Max = {max_val:.1f}, Min = {min_val:.1f}")



##################
### Visualizations
##################


import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def scatter_plot_with_reference(x, y, x_label, y_label, title, color="blue", alpha=0.6, font_size = 14):
    """Plots a scatter plot with a reference y=x line."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=alpha, color=color)
    
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Agreement")

    plt.xlabel(x_label, fontsize=font_size+2)
    plt.ylabel(y_label, fontsize=font_size+2)
    plt.title(title, fontsize=font_size)
    plt.legend(fontsize=font_size)
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

def side_by_side_boxplots(data1, data2, labels, title, x_label, y_label, font_size=14):
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
        boxprops=dict(facecolor="#6B8E23", alpha=0.8),
        medianprops=dict(color="black"),
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='darkgreen', markeredgecolor='darkgreen')
    )

    # Plot boxplots for the second dataset
    ax.boxplot(
        [data2[:, i] for i in range(n_features)],  # Extract feature columns
        positions=positions + width / 2,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="#4682B4", alpha=0.8),
        medianprops=dict(color="black"),
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='#4169E1', markeredgecolor='#4169E1')
    )
    
    # Set y-limit (increased but hidden)
    ax.set_ylim(0, 1.2)  # Set the y-limit to 1.2
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Only display these ticks

    
    ax.set_title(title, fontsize = font_size)
    ax.set_xlabel(x_label, fontsize = font_size + 2)
    ax.set_ylabel(y_label, fontsize = font_size + 2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize = font_size + 2)

    legend_handles = [
        Patch(facecolor="#4682B4", edgecolor="black", label="Test Data"),
        Patch(facecolor="#6B8E23", edgecolor="black", label="Generated Samples")
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize = font_size-1)

    plt.tight_layout()
    plt.show()



def feature_comparison_boxplots(similar_data, different_data, feature_names, title):
    """Creates feature-wise boxplots comparing similar and different prediction datasets"""
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

def compare_datasets(data1, data2, categorical_cols=None, numerical_cols=None, labels=None):
    """
    Compare two datasets

    - data1 (pd.DataFrame): The first dataset.
    - data2 (pd.DataFrame): The second dataset.
    - categorical_cols (list): List of categorical column names.
    - numerical_cols (list): List of numerical column names.
    - labels (tuple): A tuple containing labels for the datasets (label_data1, label_data2).
    """
    
    # Default labels if not provided
    if labels is None:
        labels = ('Data1', 'Data2')
    label_data1, label_data2 = labels

    # Set global font sizes
    plt.rc('font', size=14)          # Default text size
    plt.rc('axes', titlesize=16)     # Title size
    plt.rc('axes', labelsize=14)     # X and Y label size
    plt.rc('xtick', labelsize=13)    # X-axis tick size
    plt.rc('ytick', labelsize=13)    # Y-axis tick size
    plt.rc('legend', fontsize=14)    # Legend font size
    plt.rc('figure', titlesize=16)   # Figure title size
    
    # Replace inf values with NaN
    data1 = data1.replace([np.inf, -np.inf], np.nan)
    data2 = data2.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN values to avoid plotting issues
    data1 = data1.dropna()
    data2 = data2.dropna()

    # Determine columns if not provided
    if categorical_cols is None:
        categorical_cols = data1.select_dtypes(include=['object', 'category']).columns.tolist()
    if numerical_cols is None:
        numerical_cols = data1.select_dtypes(include=['number']).columns.tolist()
    
    # Compare numerical features
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data1[col], label=label_data1, fill=True, color='#8A9A5B')
        sns.kdeplot(data2[col], label=label_data2, fill=True, color='darkorange')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend(loc='upper left', frameon=True, fancybox=True)
        plt.show()

    # Compare categorical features
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        data1_counts = data1[col].value_counts(normalize=True)
        data2_counts = data2[col].value_counts(normalize=True)
        comparison_df = pd.DataFrame({label_data1: data1_counts, label_data2: data2_counts})
        comparison_df.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
        plt.title(f'Comparison of {col}')
        plt.ylabel('Proportion')
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
        
              
def compare_datasets_grid(data1, data2, numerical_cols=None, labels=None):
     """
     Compare two datasets and generate KDE plots in a grid format

     Parameters:
     - data1 (pd.DataFrame): The first dataset
     - data2 (pd.DataFrame): The second dataset
     - numerical_cols (list): column names
     - labels (tuple): A tuple containing labels for the datasets (label_data1, label_data2)
     """
     
     # Default labels if not provided
     if labels is None:
         labels = ('Generated Sensitive', 'Generated Risky')
     label_data1, label_data2 = labels

     # Set global font sizes
     plt.rc('font', size=13)
     plt.rc('axes', titlesize=16)
     plt.rc('axes', labelsize=14)
     plt.rc('xtick', labelsize=13)
     plt.rc('ytick', labelsize=13)
     plt.rc('legend', fontsize=13)
     plt.rc('figure', titlesize=16)
     
     # Replace inf values with NaN and drop NaNs
     data1 = data1.replace([np.inf, -np.inf], np.nan).dropna()
     data2 = data2.replace([np.inf, -np.inf], np.nan).dropna()

     # Determine numerical columns if not provided
     if numerical_cols is None:
         numerical_cols = data1.select_dtypes(include=['number']).columns.tolist()
     
     # Set up subplot grid
     num_plots = len(numerical_cols)
     cols = 2  # Number of columns in the grid
     rows = (num_plots // cols) + (num_plots % cols > 0)  # Calculate rows dynamically
     
     fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
     axes = axes.flatten()  # Flatten to easily iterate

     # Iterate through numerical columns and create KDE plots
     for i, col in enumerate(numerical_cols):
         sns.kdeplot(data1[col], label=label_data1, fill=True, color='#8A9A5B', ax=axes[i])
         sns.kdeplot(data2[col], label=label_data2, fill=True, color='darkorange', ax=axes[i])
         axes[i].set_title(f'Distribution of {col}')
         axes[i].set_xlabel(col)
         axes[i].set_ylabel('Density')
         axes[i].legend(loc='upper left', frameon=True, fancybox=True)

     # Remove any unused subplots
     for j in range(i + 1, len(axes)):
         fig.delaxes(axes[j])

     plt.tight_layout()
     plt.show()         
        
     
        
def compare_datasets_gridAll(original, inverted, categorical_cols=None, numerical_cols=None, labels=None):
    """
    Compare the original dataset with the inverted dataset using a grid layout.

    Parameters:
    - original (pd.DataFrame): The original dataset.
    - inverted (pd.DataFrame): The inverted dataset.
    - categorical_cols (list, optional): List of categorical column names.
    - numerical_cols (list, optional): List of numerical column names.
    - labels (tuple, optional): A tuple containing labels for the datasets (label_original, label_inverted).
    """

    # Default labels if not provided
    if labels is None:
        labels = ('Original Data', 'Generated Samples')
    label_original, label_inverted = labels

    # Set global font sizes
    plt.rc('font', size=12)          
    plt.rc('axes', titlesize=14)     
    plt.rc('axes', labelsize=12)     
    plt.rc('xtick', labelsize=11)    
    plt.rc('ytick', labelsize=12)    
    plt.rc('legend', fontsize=12)    
    plt.rc('figure', titlesize=14)   
    
    # Replace inf values with NaN and drop them
    original = original.replace([np.inf, -np.inf], np.nan).dropna()
    inverted = inverted.replace([np.inf, -np.inf], np.nan).dropna()

    # Determine columns if not provided
    if categorical_cols is None:
        categorical_cols = original.select_dtypes(include=['object', 'category']).columns.tolist()
    if numerical_cols is None:
        numerical_cols = original.select_dtypes(include=['number']).columns.tolist()

    # Select first 20 features (numerical + categorical) to match the 4x5 grid
    selected_features = numerical_cols[:20] if len(numerical_cols) >= 20 else numerical_cols + categorical_cols[:20-len(numerical_cols)]
    
    # Define grid layout: 5 rows × 4 columns
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
    axes = axes.flatten()  # Flatten the 2D grid for easy iteration

    # Store plot handles for legend
    handles, labels = [], []

    for i, col in enumerate(selected_features):
        ax = axes[i]  # Select subplot
        
        if col in numerical_cols:
            # KDE Plot for Numerical Features
            sns.kdeplot(original[col], label=label_original, fill=True, color='#8A9A5B', ax=ax)
            sns.kdeplot(inverted[col], label=label_inverted, fill=True, color='darkorange', ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
        else:
            # Bar Plot for Categorical Features
            original_counts = original[col].value_counts(normalize=True)
            inverted_counts = inverted[col].value_counts(normalize=True)
            comparison_df = pd.DataFrame({label_original: original_counts, label_inverted: inverted_counts})
            comparison_df.plot(kind='bar', ax=ax, color=['#8A9A5B', 'orange'])
            ax.set_title(f'Comparison of {col}')
            ax.set_ylabel('Proportion')
            ax.set_xlabel(col)
            ax.set_xticklabels(comparison_df.index, rotation=45)

        # Capture legend handles once for use in global legend
        if i == 0:  
            handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()  # Remove legend from individual plots

    # Add a global legend below the plots
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=14, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust layout to make space for legend

    plt.show()
    
    
def scatter_zoom(lin_test_preds, svr_test_preds, linear_preds_cnt, svr_preds_cnt):
    # Define figure and main axes
    fig, ax_main = plt.subplots(figsize=(8, 6))
    
    # Scatter plot of full dataset
    ax_main.scatter(lin_test_preds, svr_test_preds, alpha=0.7, color="#4682B4", label="Test Data")
    ax_main.scatter(linear_preds_cnt, svr_preds_cnt, alpha=0.7, color="#6B8E23", label="Generated Samples")
    
    # Add reference line
    min_val = min(lin_test_preds.min(), svr_test_preds.min(), linear_preds_cnt.min(), svr_preds_cnt.min())
    max_val = max(lin_test_preds.max(), svr_test_preds.max(), linear_preds_cnt.max(), svr_preds_cnt.max())
    ax_main.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Agreement")
    
    # Highlight region to be zoomed in
    rect_x_min, rect_x_max = 0.48, 0.63  # Define zoomed region x limits
    rect_y_min, rect_y_max = 0.45, 0.56  # Define zoomed region y limits
    ax_main.add_patch(plt.Rectangle((rect_x_min, rect_y_min), rect_x_max - rect_x_min, rect_y_max - rect_y_min,
                                    fill=False, edgecolor="black", linestyle="dashed"))
    
    ax_main.set_xlabel("Linear Regression Predictions", fontsize=14)
    ax_main.set_ylabel("SVR Predictions", fontsize=14)
    ax_main.set_title("Scatter Plot of Linear Regression vs SVR Predictions", fontsize=16)
    ax_main.legend(fontsize=12)
    ax_main.grid(alpha=0.3)
    
    # Create inset zoomed-in plot
    ax_inset = fig.add_axes([0.58, 0.15, 0.28, 0.28])  # Define position of inset
    ax_inset.scatter(lin_test_preds, svr_test_preds, alpha=0.7, color="#4682B4")
    ax_inset.scatter(linear_preds_cnt, svr_preds_cnt, alpha=0.7, color="#6B8E23")
    
    # Set zoomed-in limits
    x1, x2, y1, y2 = 0.48, 0.65, 0.46, 0.60  
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(y1, y2)
    ax_inset.grid(alpha=0.3)
    ax_inset.set_xticks([]) 
    ax_inset.set_yticks([]) 
    
    # Add reference line to inset
    ax_inset.plot([x1, x2], [x1, x2], color="red", linestyle="--")
    
    # Indicate zoom region on main plot
    ax_inset.indicate_inset_zoom(ax_inset, edgecolor="black", linestyle="dashed")
    
    plt.show()
     
        
        