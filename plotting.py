from matplotlib import pyplot as plt
import seaborn as sns

import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
from typing import Tuple, Sequence

def plot_1D_trajectory_histogram(trajectory, bins=50):
    plt.figure(figsize=(8, 6))
    
    # Plot histogram of the 1D trajectory
    plt.hist(trajectory, bins=bins, density=True, color='blue', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.title('Histogram of Trajectory')

    # Show the plot
    plt.grid(True)
    plt.show()


def plot_dots(
        X1    : jax.Array, 
        X2    : jax.Array = None,
        )-> None:
    """
    INPUT
        X1  = 2d array of shape N1 x 2 to be plotted on the 2d graph.
        X2  - 2d array of shape N2 x 2 to be plotted on the 2d graph.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(X1[:, 0], X1[:, 1], color= 'blue', alpha=0.5)
    if X2 is not None:
        plt.scatter(X2[:,0], X2[:, 1], color= 'red', alpha = 0.5)
    plt.title('Mixture of Gaussians')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.show()

def plot_lines_and_dots(
        thetas  : jax.Array,
        X1      : jax.Array = None,
        X2      : jax.Array = None,
        newX    : jax.Array = None
        )-> None:
    """
    Plots lines and points
    INPUT
        thetas  - 2d array of shape K x 3, where each row (a,b,c) corresponds to the
            line ax + by + c = 0 on the two dimensional plane.
        X1      = 2D array of shape N1 x 2 where each row (x1, y1) is a point on the
            two dimensional plane, shown in red. default is None which is then skipped
        X2      - same as X1 but depicted in red, if None then skipped.
        newX    - same as X1 and X2 but depicted in green, if none then skipped.
    """
    # Define x values for plotting
    x_vals = jnp.linspace(-7, 7, 100)

    # Create the figure
    plt.figure(figsize=(10, 10))

    for row in thetas:
        a, b, c = row
        if b != 0:
            y_vals = -(a / b) * x_vals - (c / b)
            plt.plot(x_vals, y_vals, alpha=0.1)  # Adjust alpha for transparency
        else:
            # Vertical line at x = -c/a (when b == 0)
            plt.axvline(x=-c/a, color='r', alpha=0.1)
    
    if X1 is not None:
        plt.scatter(X1[:, 0], X1[:,1], c = 'blue', alpha = 0.5)
    if X2 is not None:
        plt.scatter(X2[:, 0], X2[:,1], c = 'red', alpha = 0.5)
    if newX is not None:
        plt.scatter(newX[:,0], newX[:, 1], c = 'green', alpha = 0.1)

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundaries')
    plt.grid(True)
    plt.show()


def image_show(image, save_filename = None):
    # Display the image and label
    plt.figure(figsize=(2, 2))
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide the axes
    if save_filename is not None:
        plt.savefig(save_filename, bbox_inches = 'tight')
    plt.show()

import matplotlib.colors as mcolors

def show_filters(F): 
    A = jnp.max(jnp.abs(F))
    cmap = mcolors.TwoSlopeNorm(vmin=-A, vcenter=0, vmax=A)
    plt.imshow(F.reshape(28,28), cmap = 'bwr', norm = cmap)
    plt.colorbar(label = 'Value')
    plt.show()


from matplotlib.animation import FuncAnimation

def animate(images, save_filename = None, interval = 1, skip_over = 1):

    # Set up the figure and display the first frame
    images = images[::skip_over]
    fig, ax = plt.subplots(figsize = (2,2) )
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    im = ax.imshow(images[0], cmap='gray', interpolation= "none")  # Show the first frame with a colormap

    # Function to update the image data for each frame
    def animate(frame):
        im.set_array(images[frame])  # Update the data for imshow
        return im,

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=images.shape[0], interval=interval, blit=True)
    if save_filename is not None:
        ani.save(save_filename, writer = "pillow")
    plt.show()


def time_series(array1 : jax.Array, array2 = None, array3 = None, array4 = None):
    array1_np = np.array(array1)
    if array2 is not None:  
        array2_np = np.array(array2)
    if array3 is not None:
        array3_np = np.array(array3)
    if array4 is not None:
        array4_np = np.array(array4)
    time_indices = jnp.arange(len(array1_np))

    # Plot the time series
    plt.figure(figsize=(8, 4))
    plt.plot(time_indices, array1_np, color = 'blue', label="Array 1")
    if array2 is not None:
        plt.plot(time_indices, array2_np, color ='red', label = "Array 2")
    if array3 is not None:
        plt.plot(time_indices, array3_np, color = 'orange', label="Array 3")
    if array4 is not None:
        plt.plot(time_indices, array4_np, color ='purple', label = "Array 4")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Time Series Graph")
    plt.legend()
    plt.grid(True)
    plt.show()


def feature_comparison_histograms(
    dfs     : Sequence[pd.DataFrame], 
    n_cols  : int                  = 4, 
    figsize : Tuple[int]           = (15, 15), 
    labels  : list[str]| None      = None
    ):
    """
    Plot histograms for each feature to up do 4 dataframes.
    
    Parameters:
    -----------
    dfi : pandas.DataFrame
        Dataframe containing features
    n_cols : int, default=3
        Number of columns in the subplot grid
    figsize : tuple, default=(15, 15)
        Figure size (width, height)
    labels : list, default=['Dataset 1', 'Dataset 2']
        Labels for the datasets in the legend
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing all subplots
    """
    labels = [f'Dataset_{i}' for i in range(1, len(dfs) + 1)] if labels is None else labels        
    features = dfs[0].columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Feature Distributions', fontsize=16)
    
    # Flatten axes array for easier iteration
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Plot datasets
        for (i, dfi) in enumerate(dfs):
            sns.histplot(data=dfi, x=feature, alpha=0.5, label=labels[i], stat = 'density', ax=ax, bins = 100)
        

        ax.set_title(f'{feature}')
        ax.legend()
        
        # Rotate x-axis labels if they're too long
        if max([len(str(label)) for label in ax.get_xticklabels()]) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig

from matplotlib.gridspec import GridSpec


def create_mnist_grid(images, nrows, ncols, figsize=None):
    """
    Create a tight grid of MNIST images with arbitrary dimensions.
    
    Args:
        images: jax.Array of shape (N, 28, 28) containing MNIST images
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
        figsize: Figure size in inches. If None, automatically calculated 
                based on grid dimensions to fit double column width (6.5 inches)
    
    Returns:
        matplotlib.figure.Figure
    """
    required_images = nrows * ncols
    assert len(images) >= required_images, f"Need at least {required_images} images for {nrows}x{ncols} grid"
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        # Base size on double column width (6.5 inches)
        # Scale height proportionally to maintain square-ish cells
        width = min(6.5, 2 * ncols)  # Cap width at double column width
        height = width * (nrows / ncols)
        figsize = (width, height)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.05, wspace=0.05)
    
    # Plot each image
    for i in range(required_images):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        
        # Display image without interpolation for sharp pixels
        ax.imshow(images[i], cmap='gray', interpolation='none')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def save_mnist_grid(images, nrows, ncols, output_path='mnist_grid.png', figsize=None):
    """
    Save a grid of MNIST images.
    
    Args:
        images: jax.Array of shape (N, 28, 28)
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
        output_path: Path to save the output PNG
        figsize: Optional tuple of (width, height) in inches
    """
    fig = create_mnist_grid(images, nrows, ncols, figsize)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)





#############################
#############################
### FOR THE WINE DATSAET ####
#############################
#############################

from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

def plot_wine_features(X_train, X_test, y_train, y_test, generated_samples, 
                       feature_idx1=0, feature_idx2=1, model=None, 
                       feature_names=None, figsize=(12, 8), alpha=0.7):
    """
    Plot wine dataset results with random forest decision boundaries.
    
    Parameters:
    -----------
    X_train : array-like, shape (n_train, n_features)
        Training features
    X_test : array-like, shape (n_test, n_features) 
        Test features
    y_train : array-like, shape (n_train,)
        Training labels
    y_test : array-like, shape (n_test,)
        Test labels
    generated_samples : array-like, shape (n_generated, n_features)
        Generated samples from MALA
    feature_idx1 : int, default=0
        Index of first feature to plot (x-axis)
    feature_idx2 : int, default=1
        Index of second feature to plot (y-axis)
    model : RandomForestClassifier, optional
        Trained random forest model for decision boundary
    feature_names : list, optional
        Names of features for axis labels
    figsize : tuple, default=(12, 8)
        Figure size
    alpha : float, default=0.7
        Transparency for points
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Color scheme as requested
    # Class 0: red (dark/light), Class 1: blue (dark/light), Class 2: orange (dark/light)
    # Generated: green
    colors = {
        'train': {0: '#8B0000', 1: '#00008B', 2: '#FF8C00'},  # Dark colors for training
        'test': {0: '#FF6B6B', 1: '#6B9BFF', 2: '#FFB366'},   # Light colors for test
        'generated': '#00B300'  # Green for generated samples
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract the two features for plotting
    X_train_2d = X_train[:, [feature_idx1, feature_idx2]]
    X_test_2d = X_test[:, [feature_idx1, feature_idx2]]
    generated_2d = generated_samples[:, [feature_idx1, feature_idx2]]
    
    # Plot training data (dark colors)
    for class_label in np.unique(y_train):
        mask = y_train == class_label
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], 
                  c=colors['train'][class_label], alpha=alpha, 
                  label=f'Train Class {class_label}', s=50, edgecolors='black', linewidth=0.5)
    
    # Plot test data (light colors)
    for class_label in np.unique(y_test):
        mask = y_test == class_label
        ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
                  c=colors['test'][class_label], alpha=alpha, 
                  label=f'Test Class {class_label}', s=50, edgecolors='gray', linewidth=0.5)
    
    # Plot generated samples (green)
    ax.scatter(generated_2d[:, 0], generated_2d[:, 1], 
              c=colors['generated'], alpha=alpha*0.8, 
              label='Generated Samples', s=30, marker='^', edgecolors='darkgreen', linewidth=0.3)
    
    # Add decision boundary if RF model is provided
    if model is not None:
        # Create a mesh for decision boundary
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Extend limits slightly for better visualization
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        x_min, x_max = x_min - x_margin, x_max + x_margin
        y_min, y_max = y_min - y_margin, y_max + y_margin
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # For decision boundary, we need to create full feature vectors
        # We'll use the mean values for the other features
        mesh_points = np.zeros((xx.ravel().shape[0], X_train.shape[1]))
        mesh_points[:, feature_idx1] = xx.ravel()
        mesh_points[:, feature_idx2] = yy.ravel()
        
        # Fill other features with training data means
        for i in range(X_train.shape[1]):
            if i not in [feature_idx1, feature_idx2]:
                mesh_points[:, i] = np.mean(X_train[:, i])
        
        # Predict on mesh
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary with light contours
        boundary_colors = ['#FFE4E4', '#E4E4FF', '#FFE4CC']  # Very light versions
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], 
                   colors=boundary_colors, extend='both')
    
    # Formatting
    if feature_names is not None:
        ax.set_xlabel(f'{feature_names[feature_idx1]}', fontsize=12)
        ax.set_ylabel(f'{feature_names[feature_idx2]}', fontsize=12)
    else:
        ax.set_xlabel(f'Feature {feature_idx1}', fontsize=12)
        ax.set_ylabel(f'Feature {feature_idx2}', fontsize=12)
    
    ax.set_title('Wine Dataset: Training, Test, and Generated Samples\nwith Decision Tree Boundaries', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

from typing import Optional, List
import math
import torch

def tensor_grid_visualize(tensor_list: List[torch.Tensor], 
    titles: Optional[List[str]] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 10),
    denormalize: bool = True,
    save_path: Optional[str] = None,
    title_fontsize: int = 10,
    suptitle: Optional[str] = None,
    spacing: float = 0.05
    ) -> plt.Figure:
    """
    Create a grid visualization from a list of PyTorch tensors.
    
    Args:
        tensor_list: List of tensors, each of shape [C, H, W] where C=3 (RGB)
        titles: Optional list of titles for each image
        grid_shape: Optional (rows, cols) for grid layout. If None, auto-calculated
        figsize: Figure size as (width, height)
        denormalize: Whether to denormalize ImageNet-normalized tensors
        save_path: Optional path to save the figure
        title_fontsize: Font size for subplot titles
        suptitle: Optional main title for the entire figure
    
    Returns:
        matplotlib Figure object
    """
    if not tensor_list:
        raise ValueError("tensor_list cannot be empty")
    
    n_images = len(tensor_list)
    
    # Auto-calculate grid shape if not provided
    if grid_shape is None:
        cols = math.ceil(math.sqrt(n_images))
        rows = math.ceil(n_images / cols)
        grid_shape = (rows, cols)
    
    rows, cols = grid_shape
    
    # Validate grid can accommodate all images
    if rows * cols < n_images:
        raise ValueError(f"Grid shape {grid_shape} too small for {n_images} images")
    
    if titles:
        hspace = max(spacing * 3, 0.15)  # More vertical space for titles
    else:
        hspace = spacing
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    plt.subplots_adjust(
        left=0.02,
        bottom=0.02,
        right=0.98,
        top=0.95 if suptitle else 0.98,
        wspace=spacing,
        hspace=hspace
    )

    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # ImageNet normalization parameters for denormalization
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])
    
    for i, tensor in enumerate(tensor_list):
        if tensor.dim() not in [3, 4]:
            raise ValueError(f"Tensor {i} must be 3D [C,H,W] or 4D [B,C,H,W]")
        
        # Handle batch dimension
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Tensor {i} batch size must be 1, got {tensor.shape[0]}")
            tensor = tensor.squeeze(0)
        
        # Clone to avoid modifying original
        img_tensor = tensor.clone()
        
        # Denormalize if requested and tensor appears to be normalized
        if denormalize and tensor.min() < 0:
            # Reshape normalization parameters to match tensor dimensions
            mean = imagenet_mean.view(3, 1, 1).to(tensor.device)
            std = imagenet_std.view(3, 1, 1).to(tensor.device)
            img_tensor = img_tensor * std + mean
        
        # Clamp values to [0, 1] range for display
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to numpy and transpose for matplotlib (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Display image
        axes[i].imshow(img_np)
        axes[i].axis('off')
        
        # Add title if provided
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=title_fontsize)
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    # Add main title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=title_fontsize + 4)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

