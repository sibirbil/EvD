import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import StandardScaler


file = "./unconv_MV_v5.csv"
df = pd.read_csv(file)

X = df[["Por", "Brittle"]].values.reshape(-1, 2)
y = df["Prod"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.values.reshape(-1, 1))

x1 = X[:, 0] + np.random.normal(0, 0.01, X.shape[0])
x2 = X[:, 1] + np.random.normal(0, 0.05, X.shape[0])

meshsize = 50
x_pred = np.linspace(-1, 1, meshsize)  # range of porosity values
y_pred = np.linspace(-1, 1, meshsize)  # range of brittleness values

xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
predicted = model.predict(model_viz)

# Predictions in their original scale
# predicted = y_scaler.inverse_transform(predicted.reshape(-1, 1))

plt.style.use("default")

fig = plt.figure(figsize=(20, 15))

ax = fig.add_subplot(projection="3d")

ax.scatter(x1, x2, y, s=50, facecolors="none", edgecolors="b", alpha=0.5)

ax.plot_surface(
    xx_pred,
    yy_pred,
    predicted.reshape(meshsize, meshsize),
    cmap=cm.Spectral,
    linewidth=0,
    antialiased=False,
)
ax.set_xlabel("$x_1$", fontsize=12)
ax.set_ylabel("$x_2$", fontsize=12)
ax.set_zlabel("y", fontsize=12)

T = 10
N = X.shape[0]
D = np.column_stack((X, np.ones(X.shape[0])))
thetahat = np.linalg.inv(D.T @ D) @ D.T @ y
xbar = np.mean(X, axis=0)

yhat = D @ thetahat
ybarhat = np.mean(yhat)

covXyhat = (X - xbar).T @ (yhat - ybarhat) / (N)  #  N-1 in the denominator?
covXyhat = covXyhat.flatten()
varyhat = np.var(yhat, ddof=1)


testsize = 10

# Data infidelity
x1test = 20 + np.random.random_sample(testsize) * 5.0
x2test = 90 + np.random.random_sample(testsize) * 10.0

# Data fidelity
x1test = 13 + np.random.random_sample(testsize) * 4.0
x2test = 40 + np.random.random_sample(testsize) * 20.0


testdata = np.column_stack((x1test, x2test))
testdata = scaler.transform(testdata)

predictions = model.predict(testdata)
predictions = predictions.flatten()
# Predictions in their original scale
# predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1))

ax.scatter(
    testdata[:,0], testdata[:,1], predictions, s=50, facecolors="none", edgecolors="k", alpha=0.5
)
ax.scatter(testdata[:,0], testdata[:,1], 0.0, s=50, color="k", alpha=0.5)
for i, x0 in enumerate(testdata):
    ax.plot(
        [x0[0], x0[0]],        
        [x0[1], x0[1]],        
        [0.0, predictions[i]], 
        color="k",
        linestyle="--",
        alpha=0.2,
    )

for w in predictions:

    fhat = xbar + (covXyhat / (varyhat + T)) * (w - ybarhat)
    ksi = thetahat[:-1]
    sigmainv = T * np.linalg.inv(((X.T @ X) / N) - xbar @ xbar.T) + ksi @ ksi.T
    sigma = np.linalg.inv(sigmainv)

    x0 = np.random.multivariate_normal(mean=fhat, cov=sigma)
    ax.scatter(x0[0], x0[1], w, s=50, facecolors="none", edgecolors="r", alpha=0.5)
    ax.scatter(x0[0], x0[1], 0.0, s=50, color="r", alpha=0.5)
    ax.plot(
        [x0[0], x0[0]], [x0[1], x0[1]], [0.0, w], color="r", linestyle="--", alpha=0.2
    )

plt.show()
