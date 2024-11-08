
import LRExample
import jax
import jax.numpy as jnp
import langevin
import jax.random as random

X = LRExample.X
y = (LRExample.y).to_numpy()

X = jnp.array(X)
y = jnp.array(y)


def F_function(
    X    : jax.Array, 
    y    : jax.Array
    ):
    
    D = jnp.hstack([X, jnp.ones(X.shape[0]).reshape(-1,1)])
    
    def F(theta):
        yhat = D @ theta
        return jnp.mean(jnp.square(yhat - y))

    return F, jax.grad(F)


theta_0 = jnp.array([100.,30.,-1000.])
key = random.PRNGKey(42)
key, subkey = random.split(key)

state_theta = (theta_0, key)
F, gradF = F_function(X, y)
hypsF = (F, gradF, 0.0001) #last entry is step size

(last, last_key), traj_theta = langevin.MALA_chain(state_theta, hypsF, 1000000)

thetas = traj_theta[999900:]

def G_function(
        thetas      :jax.Array,
        y_val       :jnp.float_
):
    
    def G(x):
        d = jnp.hstack([x, 1])
        return jnp.square(jnp.mean(thetas @ d) - y_val)
    
    return G, jax.grad(G)



#testdata = jnp.array([[20, 90]])
#prediction = LRExample.model.predict(testdata)

x_0 = jnp.array([15.,45.])
state_x = (x_0, subkey)
G, gradG = G_function(thetas, 3852.1)
hypsG = G, gradG, 0.00001

_, traj_x = langevin.MALA_chain(state_x, hypsG, 1000000)

last_xs = traj_x[-100:]



