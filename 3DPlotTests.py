import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


file = "./unconv_MV_v5.csv"
df = pd.read_csv(file)

X = df[["Por", "Brittle"]].values.reshape(-1, 2)
y = df["Prod"]


x1 = X[:, 0] + np.random.normal(0, 3, X.shape[0])
x2 = X[:, 1] + np.random.normal(0, 10, X.shape[0])

meshsize = 50
x_pred = np.linspace(6, 24, meshsize)  # range of porosity values
y_pred = np.linspace(0, 100, meshsize)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
predicted = model.predict(model_viz)

plt.style.use("default")

fig = plt.figure(figsize=(20, 15))

ax = fig.add_subplot(projection="3d")

ax.plot(x1, x2, y, color="b", zorder=15, linestyle="none", marker="o", alpha=0.5)
ax.plot_surface(
    xx_pred,
    yy_pred,
    predicted.reshape(meshsize, meshsize),
    cmap=cm.viridis,
    linewidth=0,
    antialiased=False,
)
ax.set_xlabel("$x_1$", fontsize=12)
ax.set_ylabel("$x_2$", fontsize=12)
ax.set_zlabel("y", fontsize=12)
ax.set_title("Give me some favorite samples that have output between 4000 and 5000")
# ax.locator_params(nbins=4, axis="x")
# ax.locator_params(nbins=5, axis="x")


T = 1000
N = X.shape[0]
D = np.column_stack((X, np.ones(X.shape[0])))
# checkthetathat = np.linalg.lstsq(D, y, rcond=None)[0]
thetahat = np.linalg.inv(D.T @ D) @ D.T @ y
xbar = np.mean(X, axis=0)

yhat = D @ thetahat
ybarhat = np.mean(yhat)

# covXyhat = (X - xbar).T @ (yhat - ybarhat) / (X.shape[0] - 1) # denominatorda -1 olmayacak mı?
covXyhat = (X - xbar).T @ (yhat - ybarhat) / (N)
varyhat = np.var(yhat, ddof=1)


batchsize = 30
for _ in range(batchsize):

    w = 4000 + (np.random.rand() * 1000)

    fhat = xbar + (covXyhat / (varyhat + T)) * (w - ybarhat)
    ksi = thetahat[:-1]
    sigmainv = T * np.linalg.inv(((X.T @ X) / N) - xbar @ xbar.T) + ksi @ ksi.T
    sigma = np.linalg.inv(sigmainv)

    x0 = np.random.multivariate_normal(mean=fhat, cov=sigma)
    ax.plot(
        x0[0], x0[1], w, color="r", zorder=15, linestyle="none", marker="s", alpha=0.5
    )

plt.show()

print()
