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

def create_2x2_grid(images, figsize=(3.25, 3.25)):
    """
    Create a tight 2x2 grid of MNIST images.
    
    Args:
        images: jax.Array of shape (4, 28, 28) containing MNIST images
        figsize: Figure size in inches (default is half of double column width)
    
    Returns:
        matplotlib.figure.Figure
    """
    assert len(images) == 4, "Exactly 4 images required"
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.05)
    
    # Plot each image
    for i in range(4):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Display image without interpolation for sharp pixels
        ax.imshow(images[i], cmap='gray', interpolation='none')
        ax.axis('off')
        
            
    plt.tight_layout()
    return fig

def save_2x2_grid(images, output_path='mnist_2x2.png'):
    """
    Save a 2x2 grid of MNIST images.
    
    Args:
        images: jax.Array of shape (4, 28, 28)
        output_path: Path to save the output PNG
    """
    fig = create_2x2_grid(images)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)