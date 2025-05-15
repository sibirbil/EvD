from sklearn.svm import SVC
from sklearn.datasets import make_circles
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from jax import random as random
from typing import Callable, Optional
import langevin

gamma = 20.

X,y  = make_circles((50,50), noise =0.05, factor = 0.5, random_state = 42)

clf1 = SVC(kernel = 'rbf', C = 10., gamma = gamma)
clf1.fit(X,y)

clf2 = SVC(kernel = 'poly', degree= 3, gamma =gamma, C = 10, coef0 = 1.)
clf2.fit(X,y)

def rbf_kernel(x, support_vecs, gamma):
    # Use squared norm directly - more stable than computing then squaring
    squared_diffs = jnp.sum(jnp.square(x - support_vecs), axis=1)
    # Clip extremely large negative values in the exponent to prevent underflow
    exponent = jnp.clip(-gamma * squared_diffs, -15.0, 0.0)
    
    return jnp.exp(exponent)
    
def poly_kernel(x, support_vecs, degree, gamma, coef0):
    return jnp.power(gamma*jnp.dot(support_vecs, x) + coef0, degree)

def decision_function(
    clf:SVC, 
    gamma:float
    ):
    yalpha = jnp.array(clf.dual_coef_[0])
    xs = jnp.array(clf.support_vectors_)
    b = clf.intercept_[0]
    if clf.kernel == 'rbf':
        def decision_fn(x):
            pred_fn = jnp.sum(yalpha*rbf_kernel(x, xs, gamma)) + b
            return pred_fn 
    if clf.kernel == 'poly':
        coef0 = clf.coef0
        degree = clf.degree
        def decision_fn(x):
            pred_fn= jnp.sum(yalpha*poly_kernel(x,xs, degree, gamma, coef0)) + b
            return pred_fn
    
    return decision_fn
    
decision1 = decision_function(clf1, gamma)
decision2 = decision_function(clf2, gamma)

def show_regions(X, y,clf:SVC, extras = None):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict class labels
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and training points
    # plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap = 'bwr')
    # plt.scatter(X[69,0], X[69,1], c = 'orange', marker = 'o', s = 100)
    #plotting 
    #plt.scatter(X[clf.support_, 0], X[clf.support_, 1], c='red', marker='*', s=80)
    if extras is not None:
        plt.scatter(extras[:, 0], extras[:,1], c = 'green', s = 2, alpha = 0.1)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

def make_grid(name:Optional[str] = None)->None:
    
    fig, axes = plt.subplots(1,4, figsize = (6.5, 1.625))
    fig.subplots_adjust(wspace = 0.1)
    plt.rcParams['font.size'] = 9  # Match paper font size
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    for i, ax in enumerate(axes):
        ax.scatter(X[:, 0],X[:,1], c = y, edgecolors = 'k', marker = 'o', cmap = 'bwr', s = 10)  
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        subplot_label = f'({chr(97+i)})'   #(a). (b), (c), and (d)
        ax.text(0.02, 0.98,subplot_label, transform = ax.transAxes, fontsize = 9, fontweight = 'normal', va= 'top')
        
    axes[0].scatter(traj_p_contrast[-1000:,0], traj_p_contrast[-1000:,1], c = 'green', s= 2, alpha = 0.1)
    axes[1].scatter(traj_p_risky1[:,0], traj_p_risky1[:,1], c = 'green', s= 2, alpha = 0.1)
    axes[2].scatter(traj_p_risky2[:,0], traj_p_risky2[:,1], c = 'green', s= 2, alpha = 0.1)
    axes[3].scatter(traj_p_fixed[-4000:,0], traj_p_fixed[-4000:,1], c = 'green', s= 2, alpha = 0.1)
    axes[3].scatter(X[69,0], X[69,1], c = 'orange', marker = 'o', s = 20)
    
    plt.tight_layout()
    if name is not None:
        fig.savefig(name, format='pdf', bbox_inches='tight', dpi=300)

    plt.show()


###########
## CONTRAST POINTS
###########

def G_function_contrast(
    decision1 :Callable,
    decision2 :Callable,
    beta :float,
    l2_reg : float
    ):
    
    def G(x):
        x_copy = jax.lax.stop_gradient(x)
        y = - jnp.sign(decision2(x_copy))  #opposite label
        loss = jax.nn.relu(1 - y*decision1(x))
        return beta*loss + l2_reg*jnp.dot(x,x)
    
    return G


betaG_contrast = 10.
l2_contrast = 10.
etaG_contrast = 0.05/betaG_contrast
p0 = jnp.array([0., 0.])

G_contrast = G_function_contrast(decision1, decision2, betaG_contrast, l2_contrast)
hypsG_contrast = G_contrast, jax.grad(G_contrast), etaG_contrast
keyMALA_contrast = random.PRNGKey(41)
keyp0_contrast = random.PRNGKey(141)
p0_contrast = random.truncated_normal(keyp0_contrast, -1., 1., (2,) )
state_p_contrast = keyMALA_contrast, p0_contrast

_, traj_p_contrast = langevin.MALA_chain(state_p_contrast, hypsG_contrast, 100000)


########
## RISKY POINTS
########

def G_function_risky(
    decision_fn : Callable,
    beta        : float,
    l2_reg      : float,
    anchor      : jax.Array = jnp.array([0.,0.]),
    r           : float = 1 # rnorm
    ):

    def G(x):
        loss = jnp.pow(jnp.abs(decision_fn(x)),r) # when decision is 0
        localizer = jnp.sum(jnp.square(x - anchor))
        return beta*loss + l2_reg*localizer
    
    return G

betaG_risky = 100.
etaG_risky = 0.001/betaG_risky
l2_risky = 30.
G_risky1 = G_function_risky(decision1, betaG_risky, l2_risky)
G_risky2 = G_function_risky(decision2, betaG_risky, l2_risky)

def combine_trajs(G):
    keyp0_risky = random.PRNGKey(142)
    keys_ps = random.split(keyp0_risky, 100)
    hypsG = G, jax.grad(G), etaG_risky
    result = []
    for key in keys_ps:
        key, MALA_key = random.split(key)
        state_p = MALA_key, random.truncated_normal(key, -0.5, 0.5, (2,))
        _, traj_p = langevin.MALA_chain(state_p, hypsG, 2000)
        result.append(traj_p[-1000:])
    return jnp.concatenate(result, axis = 0)

traj_p_risky1 = combine_trajs(G_risky1)
traj_p_risky2 = combine_trajs(G_risky2)

########
## FIXED LABEL
########

def G_function_fixed_label(
    decision_fn :Callable,
    beta : float, #inverse temperature
    l2_reg : float,
    y : int, # + 1 or - 1 
    anchor = jnp.array([0., 0.])
    ):
    
    def G(x):
        loss = jax.nn.relu(1 - y*decision_fn(x))
        localizer = jnp.sum(jnp.square(x - anchor))
        return beta*loss + l2_reg*localizer
    
    return G


betaG_fixed = 100.
etaG_fixed  = 0.001/betaG_fixed
l2_fixed = 10.
label_fixed = -1

p0_fixed = jnp.array(X[69])
keyMALA_fixed = random.PRNGKey(42)
state_p = keyMALA_fixed, p0_fixed

G = G_function_fixed_label(decision1, betaG_fixed, l2_fixed, y = label_fixed, anchor = p0_fixed)

hypsG = G, jax.grad(G), etaG_fixed
_, traj_p_fixed = langevin.MALA_chain(state_p, hypsG, 10000)



