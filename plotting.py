from matplotlib import pyplot as plt
import jax.numpy as jnp
import jax


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
    x_vals = jnp.linspace(-5, 5, 100)

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

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundaries')
    plt.grid(True)
    plt.show()

