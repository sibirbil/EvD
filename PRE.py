#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:11:02 2024

@author: u1573378
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Load data
file = "./unconv_MV_v5.csv"
df = pd.read_csv(file)

# Features and target
X = df[["Por", "Brittle"]].values
y = df["Prod"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create polynomial features
degree = 3  # Degree of the polynomial
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)
X_poly = X_poly[:,1:]

# Train polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Create prediction grid
meshsize = 50
x_pred = np.linspace(-1, 1, meshsize)  # Range of porosity values
y_pred = np.linspace(-1, 1, meshsize)  # Range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
model_viz_poly = poly.transform(model_viz)
model_viz_poly = model_viz_poly[:,1:] 

# Make predictions
predicted = model.predict(model_viz_poly)

# Visualization setup
plt.style.use("default")
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(projection="3d")

# Add noise for visualization
x1 = X[:, 0] + np.random.normal(0, 0.01, X.shape[0])
x2 = X[:, 1] + np.random.normal(0, 0.05, X.shape[0])

# Plot original data points
ax.scatter(x1, x2, y, s=50, facecolors="none", edgecolors="b", alpha=0.5)

# Plot polynomial regression surface
ax.plot_surface(
    xx_pred,
    yy_pred,
    predicted.reshape(meshsize, meshsize),
    cmap=cm.Spectral,
    linewidth=0,
    antialiased=False,
)

# Labels
ax.set_xlabel("$x_1$", fontsize=12)
ax.set_ylabel("$x_2$", fontsize=12)
ax.set_zlabel("y", fontsize=12)


T = 1000
N = X.shape[0]
D = np.column_stack((X_poly, np.ones(X_poly.shape[0])))  # Add a column of ones for the intercept
thetahat = np.linalg.inv(D.T @ D) @ D.T @ y  # Calculate regression coefficients
xbar = np.mean(X_poly, axis=0)  # Mean of features

yhat = D @ thetahat
ybarhat = np.mean(yhat)

covXyhat = (X_poly - xbar).T @ (yhat - ybarhat) / (N)  #  N-1 in the denominator?
varyhat = np.var(yhat, ddof=1)

testsize = 10

# Data infidelity
x1test = 10 + np.random.random_sample(testsize) * 3.0
x2test = 30 + np.random.random_sample(testsize) * 5.0

# Data fidelity
# x1test = 13 + np.random.random_sample(testsize) * 4.0
# x2test = 40 + np.random.random_sample(testsize) * 20.0


testdata = np.column_stack((x1test, x2test))
testdata = scaler.transform(testdata)

testdata_poly = poly.transform(testdata)  # Transform test data to polynomial features
testdata_poly = testdata_poly[:,1:]
predictions = model.predict(testdata_poly)


ax.scatter(
    testdata[:,0], testdata[:,1], predictions, s=50, facecolors="none", edgecolors="k", alpha=0.5
)
ax.scatter(testdata[:,0], testdata[:,1], 0.0, s=50, color="k", alpha=0.5)
i = 0
for x0 in testdata:
    ax.plot(
        [x0[0], x0[0]],
        [x0[1], x0[1]],
        [0.0, predictions[i]],
        color="k",
        linestyle="--",
        alpha=0.6,   # Increased opacity
        linewidth=1.5  # Thicker lines
    )
    i += 1


for w in predictions:
    fhat = xbar + (covXyhat / (varyhat + T)) * (w - ybarhat)
    ksi = thetahat[:-1]
    sigmainv = T * np.linalg.inv(((X_poly.T @ X_poly) / N) - xbar @ xbar.T) + ksi @ ksi.T
    sigma = np.linalg.inv(sigmainv)

    x0 = np.random.multivariate_normal(mean=fhat, cov=sigma)
    ax.scatter(x0[0], x0[1], w, s=50, facecolors="none", edgecolors="r", alpha=0.5)
    ax.scatter(x0[0], x0[1], 0.0, s=50, color="r", alpha=0.5)
    ax.plot(
        [x0[0], x0[0]], [x0[1], x0[1]], [0.0, w],
        color="r",
        linestyle="--",
        alpha=0.6,   # Increased opacity
        linewidth=1.5  # Thicker lines
    )

fig.show()


"""

# Initialize the Plotly figure
fig = go.Figure()

# Polynomial regression surface
fig.add_trace(go.Surface(
    z=predicted.reshape(meshsize, meshsize),
    x=xx_pred,
    y=yy_pred,
    colorscale='Spectral',
    opacity=0.8,
    name='Regression Surface'
))

# Scatter plot for original data points with noise
fig.add_trace(go.Scatter3d(
    x=x1, y=x2, z=y,
    mode='markers',
    marker=dict(size=4, color='blue', opacity=0.5),
    name='Original Data'
))

# Data fidelity and infidelity points from test data
for i, w in enumerate(predictions):
    # Ensure `fhat` and `sigma` have matching dimensions with full polynomial feature space
    #fhat = np.mean(D, axis=0) + (covXyhat / (varyhat + T)) * (w - ybarhat)
    
    # Calculate fhat using the reshaped covXyhat for consistency
    fhat = np.mean(D, axis=0) + (covXyhat / (varyhat + T)).reshape(-1, 1) * (w - ybarhat)

    
    ksi = thetahat  # Using full thetahat for consistency in dimension
    sigmainv = T * np.linalg.inv(((D.T @ D) / N) - np.mean(D, axis=0) @ np.mean(D, axis=0).T) + ksi @ ksi.T
    sigma = np.linalg.inv(sigmainv + 1e-3 * np.eye(sigmainv.shape[0]))  # Regularization added to sigma

    try:
        # Generate sample point based on corrected dimensions
        x0 = np.random.multivariate_normal(mean=fhat, cov=sigma)

        # Plot generated points and connecting lines
        fig.add_trace(go.Scatter3d(
            x=[x0[0]], y=[x0[1]], z=[w],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.5),
            name='Generated Point'
        ))
        fig.add_trace(go.Scatter3d(
            x=[x0[0], x0[0]], y=[x0[1], x0[1]], z=[0.0, w],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ))
    except ValueError:
        # Skip this point if dimension mismatch occurs
        print("Skipped point due to dimension mismatch.")

# Configure layout for Plotly
fig.update_layout(
    scene=dict(
        xaxis_title="Porosity",
        yaxis_title="Brittleness",
        zaxis_title="Production"
    ),
    title="Interactive 3D Polynomial Regression Surface with Fidelity/Infidelity Points",
    autosize=True
)

# Display the interactive Plotly figure

"""
