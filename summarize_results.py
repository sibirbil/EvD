
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def summary_plots(factual, counterfactual_df: pd.DataFrame,  
                                   numerical_features: List[str], categorical_features: List[str]):
    """
    Visualizes the distributions of numerical and categorical features between factual and counterfactual instances in a single grid.
    
    Parameters:
        df (pd.DataFrame): Original dataset containing the factual data.
        counterfactual_df (pd.DataFrame): DataFrame containing counterfactual samples.
        row_number (int): Row number in the original dataset to extract factual values.
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
    """
    factual_color = "#E63946"  # Replace with the extracted red color
    cnt_color = "#457B9D"  # Replace with the extracted blue color

    # Create a grid for all 9 plots (3x3)
    total_features = len(numerical_features) + len(categorical_features)
    rows = (total_features // 3) + (1 if total_features % 3 != 0 else 0)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows), constrained_layout=True)
    axes = axes.flatten()

    # Plot Numerical Features
    for i, feature in enumerate(numerical_features):
        sns.kdeplot(counterfactual_df[feature], fill=True, alpha=0.5, color= cnt_color, label='Counterfactuals', ax=axes[i])
        axes[i].axvline(factual[feature], color= factual_color, linestyle='--', label=f'Factual ({factual[feature]})')
        axes[i].set_title(f"{feature.replace('-', ' ').title()} Distribution")
        axes[i].set_xlabel(feature.replace('-', ' ').title())
        axes[i].set_ylabel('Density')
        # Swap legend order
        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].legend(handles=[handles[1], handles[0]], labels=[labels[1], labels[0]])

        # Set grid for better readability
        axes[i].grid(alpha=0.3)

        # Swap legend order and update style
        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].legend(
            handles=[handles[1], handles[0]],  # Swap factual and counterfactual order
            labels=[labels[1], labels[0]],
            loc="upper left",  # Move legend to the left
            fontsize=9,
            frameon=True,
            fancybox=True,
            framealpha=0.7  # Add transparency to the legend background
            )    


    # Plot Categorical Features
    results = {}
    for feature in categorical_features:
        factual_value = factual[feature]
        total_count = len(counterfactual_df)

        # Count occurrences of each category
        category_counts = counterfactual_df[feature].value_counts(normalize=True) * 100

        # Ensure all categories are included, even if absent in counterfactuals
        if factual_value not in category_counts:
            category_counts[factual_value] = 0

        # Sort categories alphabetically
        category_counts = category_counts.sort_index()

        # Store the results for visualization
        results[feature] = category_counts

    for j, (feature, percentages) in enumerate(results.items(), start=len(numerical_features)):
        # Highlight the factual category in red and counterfactuals in blue
        factual_value = factual[feature]

        # Adjust bar width for single bars and add padding for better visualization
        if len(percentages) == 1:
            axes[j].bar([0], percentages.values, color=cnt_color, alpha=0.7, width=0.2)
            axes[j].set_xticks([0])
            axes[j].set_xlim(-0.5, 0.5)
            axes[j].set_xticklabels(percentages.index)
        else:
            axes[j].bar(percentages.index, percentages.values, color=cnt_color, alpha=0.7, width=0.5)


        # Highlight the factual category by changing its text color to red and adding a star
        xtick_labels = []
        tick_colors = []
        for category in percentages.index:
            if category == factual_value:
                xtick_labels.append(f"*{category}")  # Add a star to the label
                tick_colors.append(factual_color)  # Red color for factual
            else:
                xtick_labels.append(category)
                tick_colors.append("black")  # Default black color for others

        # Update the x-tick labels
        axes[j].set_xticks(range(len(percentages.index)))
        axes[j].set_xticklabels(xtick_labels, fontsize=10, rotation=45)

        # Manually set the color for each x-tick
        for tick, color in zip(axes[j].xaxis.get_major_ticks(), tick_colors):
            tick.label1.set_color(color)
            
        # Add gridlines to the background
        axes[j].grid(axis="y", linestyle="--", alpha=0.7)  # Add gridlines to the y-axis
        
        
        axes[j].set_title(f"{feature.title()} Distribution", fontsize=12, fontweight='bold')
        axes[j].set_ylabel("Percentage", fontsize=10)
        axes[j].set_xlabel(feature.title(), fontsize=10)
        axes[j].tick_params(axis="both", which="major", labelsize=9)


        # Add legend for factual and counterfactuals
        factual_patch = plt.Line2D([0], [0], color=factual_color, lw=4, label=f'Factual ({factual_value})')
        counterfactual_patch = plt.Line2D([0], [0], color=cnt_color, lw=4, label='Counterfactuals')
        axes[j].legend(handles=[factual_patch, counterfactual_patch], loc="upper left", fontsize = 9)

    # Remove unused axes if any
    for k in range(len(numerical_features) + len(categorical_features), len(axes)):
        fig.delaxes(axes[k])

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
    
    