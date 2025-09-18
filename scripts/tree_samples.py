import sklearn.datasets as datasets
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import entropy
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_wine_features

wine = load_wine()
X, y = wine.data, wine.target

scaler = MaxAbsScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 52)

rf =  RandomForestClassifier(
    n_estimators = 100,
    max_depth = 3,
    min_samples_leaf = 10,
    min_samples_split = 40,
    max_features='sqrt',
    bootstrap= True,
    oob_score=True,
)

rf.fit(X,y)

dt = DecisionTreeClassifier(
    max_depth = None,
    min_samples_split=20,
    min_samples_leaf=5
    )

dt.fit(X,y)

def predict_in_dt_high_entropy_in_rf(x):
    neg_entropy = - entropy(rf.predict_proba(x), axis = 1)
    class_anchor = np.square(dt.predict(x) - 1)
    return neg_entropy + class_anchor

import langevin

hyps = langevin.Hyperparameters(
    func = predict_in_dt_high_entropy_in_rf,
    beta = 1000.,
    eta = 0.01,
    sigma = 0.05,
    reg = lambda x: np.where(np.all((x > 0) & (x < 1), axis =1), 0, np.inf)
    )

x_init = np.random.random((X[0:1].shape))
x_last, traj = langevin.adaptive_np_MALA(x_init,hyps, 1000)


fig, ax = plot_wine_features(
            X_train=X_train,
            X_test=X_test, 
            y_train=y_train,
            y_test=y_test,
            generated_samples=traj[500:],
            feature_idx1=11,
            feature_idx2=12,
            model= dt,
            feature_names= wine.feature_names
    )
    

plt.show()
