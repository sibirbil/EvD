
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import ast
import numpy as np
from scipy.stats import gaussian_kde
from collections import defaultdict


def read_sample(file_path):
    """
    Read the sample index and the bounds from a CSV file
    Returns:
        pd.DataFrame: DataFrame with 'record_index', 'lower_bounds', and 'upper_bounds'
    """
    
    sample_info = pd.read_csv(file_path)
    
    # Convert string representation of lists to actual lists
    sample_info["lower_bounds"] = sample_info["lower_bounds"].apply(ast.literal_eval)
    sample_info["upper_bounds"] = sample_info["upper_bounds"].apply(ast.literal_eval)
    
    return sample_info


from textwrap import fill

LABEL_ALIAS = {
    "government": "govt.",
    "Exec-managerial": "Exec-mng"
    # "self-employed": "self-empl.",
    # "private": "priv."
}

def _wrap_xticklabels(ax, width=8, rotation=45, ha="right", pad=6, size=10):
    labels = [fill(t.get_text(), width=width) for t in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=size, rotation=rotation, pad=pad)



def summary_plots_with_error(
    factual: dict,
    counterfactual_runs: List[pd.DataFrame],
    numerical_features: List[str],
    categorical_features: List[str],
    wrap_width: int = 10,
    rotate_deg: int = 45,
):
    
    factual_color = "#E63946"
    cnt_color = "#457B9D"

    total_features = len(numerical_features) + len(categorical_features)
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), constrained_layout=False)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.8, wspace=0.35)

    factual_patch = plt.Line2D([0], [0], color=factual_color, linestyle='--', lw=3, label='Factual')
    counterfactual_patch = plt.Rectangle((0, 0), 1, 1, color=cnt_color, alpha=0.7, label='Counterfactuals')

    # ---------- Numerical ----------
    for i, feature in enumerate(numerical_features):
        # collect KDEs across runs
        all_vals = [df[feature].dropna().values for df in counterfactual_runs if feature in df]
        if not all_vals:
            fig.delaxes(axes[i]); continue

        xmin = min(np.min(v) for v in all_vals)
        xmax = max(np.max(v) for v in all_vals)
        x_grid = np.linspace(xmin, xmax, 200)

        kde_vals = []
        for v in all_vals:
            kde = gaussian_kde(v)
            kde_vals.append(kde(x_grid))
        kde_vals = np.array(kde_vals)
        mean_kde = kde_vals.mean(axis=0)
        std_kde = kde_vals.std(axis=0)

        ax = axes[i]
        ax.plot(x_grid, mean_kde, color=cnt_color)
        ax.fill_between(x_grid, mean_kde - std_kde, mean_kde + std_kde, color=cnt_color, alpha=0.3)
        ax.axvline(factual[feature], color=factual_color, linestyle='--')
        ax.set_title(f"{feature.replace('-', ' ').title()} Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel(feature.replace('-', ' ').title(), fontsize=15)
        ax.set_ylabel('Density', fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(alpha=0.3)

    # ---------- Categorical ----------
    for j, feature in enumerate(categorical_features, start=len(numerical_features)):
        ax = axes[j]
        factual_value = factual[feature]

        all_counts = []
        all_cats = set()
        for df in counterfactual_runs:
            if feature not in df:
                continue
            counts = df[feature].value_counts(normalize=True) * 100
            all_counts.append(counts)
            all_cats.update(counts.index)

        if not all_counts:
            fig.delaxes(ax); continue

        all_cats = sorted(all_cats)

        means, stds = [], []
        for cat in all_cats:
            vals = [counts.get(cat, 0.0) for counts in all_counts]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        if len(all_cats) == 1:
            ax.bar([0], means, yerr=stds, capsize=5, color=cnt_color, alpha=0.7, width=0.25)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([0])
        else:
            ax.bar(range(len(all_cats)), means, yerr=stds, capsize=5, color=cnt_color, alpha=0.7)

        
        xtick_labels, tick_colors = [], []
        for cat in all_cats:
            alias = LABEL_ALIAS.get(str(cat), str(cat))
            if cat == factual_value:
                xtick_labels.append(f"*{alias}")
                tick_colors.append(factual_color)
            else:
                xtick_labels.append(alias)
                tick_colors.append("black")

        ax.set_xticks(range(len(all_cats)))
        ax.set_xticklabels(xtick_labels)
        for tick, color in zip(ax.xaxis.get_major_ticks(), tick_colors):
            tick.label1.set_color(color)
        _wrap_xticklabels(ax, width=wrap_width, rotation=rotate_deg, ha="right", pad=8, size=12)

        ax.set_title(f"{feature.title()} Distribution", fontsize=16, fontweight='bold')
        ax.set_ylabel("Percentage", fontsize=15)
        ax.set_xlabel(feature.title(), fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # remove unused axes
    for k in range(total_features, len(axes)):
        fig.delaxes(axes[k])

    
    fig.legend(handles=[factual_patch, counterfactual_patch],
               loc='lower center', ncol=2, fontsize=14,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.10, 1, 1])  # more bottom space for legend
    plt.show()



def summary_plots(factual, counterfactual_df: pd.DataFrame,  
                                   numerical_features: List[str], categorical_features: List[str]):

    
    factual_color = "#E63946"  
    cnt_color = "#457B9D"  

    total_features = len(numerical_features) + len(categorical_features)
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), constrained_layout=False)
    axes = axes.flatten()
    
    # spacing between subplots
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    
    # legend handles
    factual_patch = plt.Line2D([0], [0], color=factual_color, linestyle='--', lw=3, label='Factual')
    counterfactual_patch = plt.Rectangle((0, 0), 1, 1, color=cnt_color, alpha=0.7, label='Counterfactuals')


    # Numerical Features
    for i, feature in enumerate(numerical_features):
        sns.kdeplot(counterfactual_df[feature], fill=True, alpha=0.5, color= cnt_color, ax=axes[i])
        axes[i].axvline(factual[feature], color= factual_color, linestyle='--')
        axes[i].set_title(f"{feature.replace('-', ' ').title()} Distribution", fontsize = 16, fontweight='bold')
        axes[i].set_xlabel(feature.replace('-', ' ').title(), fontsize = 15)
        axes[i].set_ylabel('Density', fontsize = 15)
        axes[i].tick_params(axis="both", which="major", labelsize=14)
        axes[i].grid(alpha=0.3)


    # Categorical Features
    results = {}
    for feature in categorical_features:
        factual_value = factual[feature]
        total_count = len(counterfactual_df)

        # occurrences of each category
        category_counts = counterfactual_df[feature].value_counts(normalize=True) * 100

        # Ensure all categories are included, even if absent in counterfactuals
        if factual_value not in category_counts:
            category_counts[factual_value] = 0

        # sort categories alphabetically
        category_counts = category_counts.sort_index()
        results[feature] = category_counts

    for j, (feature, percentages) in enumerate(results.items(), start=len(numerical_features)):
        # factual category in red and counterfactuals in blue
        factual_value = factual[feature]

        # add padding for better visualization
        if len(percentages) == 1:
            axes[j].bar([0], percentages.values, color=cnt_color, alpha=0.7, width=0.2)
            axes[j].set_xticks([0])
            axes[j].set_xlim(-0.5, 0.5)
            axes[j].set_xticklabels(percentages.index, fontsize = 14)
        else:
            axes[j].bar(percentages.index, percentages.values, color=cnt_color, alpha=0.7, width=0.5)


        # Highlight the factual category by changing its text color to red and adding a star
        xtick_labels = []
        tick_colors = []
        for category in percentages.index:
            alias = LABEL_ALIAS.get(str(category), str(category))
            if category == factual_value:
                xtick_labels.append(f"*{alias}")
                tick_colors.append(factual_color)
            else:
                xtick_labels.append(alias)
                tick_colors.append("black")  # Default black color for others

        # Update the x-tick labels
        axes[j].set_xticks(range(len(percentages.index)))
        axes[j].set_xticklabels(xtick_labels, fontsize=14)


        for tick, color in zip(axes[j].xaxis.get_major_ticks(), tick_colors):
            tick.label1.set_color(color)
            
        # sdd gridlines 
        axes[j].grid(axis="y", linestyle="--", alpha=0.7)  # Add gridlines to the y-axis
        
        
        axes[j].set_title(f"{feature.title()} Distribution", fontsize=16, fontweight='bold')
        axes[j].set_ylabel("Percentage", fontsize=15)
        axes[j].set_xlabel(feature.title(), fontsize=15)
        axes[j].tick_params(axis="both", which="major", labelsize=14)

    # Remove unused axes if any
    for k in range(total_features, len(axes)):
        fig.delaxes(axes[k])

    # Add a single global legend below the figure
    fig.legend(handles=[factual_patch, counterfactual_patch], loc='lower center', ncol=2, fontsize=14, frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))

    # Adjust layout to fit legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    


def visualize_samples(subset_data, gender_column="gender", features_top=None, features_bottom=None):
    
    """
    Visualizes sample dataset.

    Parameters:
    - subset_data: The dataset to visualize.
    - gender_column (str): The name of the column indicating gender.
    - features_top (list): List of numeric features for box plots (default: common features).
    - features_bottom (list): List of categorical features for bar plots (default: common features).
    """

    # color palette for genders
    gender_palette_neutral = {"Female": "lightblue", "Male": "lightcoral"}

    # Default features if none are provided
    if features_top is None:
        features_top = ["age", "hours-per-week", "educational-num"]
    if features_bottom is None:
        features_bottom = ["race", "native-country", "workclass"]

    # Create a grid layout for the plots
    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(20, 20), gridspec_kw={"height_ratios": [1, 2]}
    )

    # Define font sizes
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize = 12

    # Top row: Box plots for selected features with gender as the y-axis
    for i, feature in enumerate(features_top):
        sns.boxplot(
            data=subset_data,
            x=feature,
            y=gender_column,
            palette=gender_palette_neutral,
            ax=axes[0, i]
        )
        axes[0, i].set_title(f"{feature.title()} vs {gender_column.title()}", fontsize=title_fontsize)
        axes[0, i].set_ylabel(gender_column.title(), fontsize=label_fontsize)
        axes[0, i].set_xlabel(feature.title(), fontsize=label_fontsize)
        axes[0, i].tick_params(axis="both", labelsize=tick_fontsize)

    # Bottom row: Bar plots for categorical features
    for i, feature in enumerate(features_bottom):
        sns.countplot(
            data=subset_data,
            x=feature,
            hue=gender_column,
            dodge=True,  # Adds space between bars for different genders
            palette=gender_palette_neutral,
            ax=axes[1, i]
        )
        axes[1, i].set_title(f"{feature.title()} Distribution by {gender_column.title()}", fontsize=title_fontsize)
        axes[1, i].set_xlabel(feature.title(), fontsize=label_fontsize)
        axes[1, i].set_ylabel("Count", fontsize=label_fontsize)
        axes[1, i].tick_params(axis="x", rotation=45, labelsize=tick_fontsize)
        axes[1, i].tick_params(axis="y", labelsize=tick_fontsize)

    plt.tight_layout() 
    
    