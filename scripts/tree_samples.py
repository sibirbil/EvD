import sklearn.datasets as datasets
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from scipy.stats import entropy
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_wine_features

wine = load_wine()
X_orig, y_orig = wine.data, wine.target

scaler = MinMaxScaler()
X = scaler.fit_transform(X_orig)

X_train, X_test, y_train, y_test = train_test_split(X,y_orig, test_size = 0.1, random_state = 52)

rf =  RandomForestClassifier(
    n_estimators = 100,
    max_depth = 3,
    min_samples_leaf = 10,
    min_samples_split = 40,
    max_features='sqrt',
    bootstrap= True,
    oob_score=True,
    random_state=641
)

rf.fit(X_train,y_train)

dt = DecisionTreeClassifier(
    max_depth = None,
    min_samples_split=20,
    min_samples_leaf=5
    )
dt.fit(X_train,y_train)

clf = XGBClassifier(
    eval_metric= 'mlogloss',
    objective = 'multi:softprob',
    n_estimators = 10,
    max_depth = 3, 
    learning_rate = 0.5, 
    random_state= 35,
    subsample = 0.8,
    colsample_bytree = 0.8,
    gamma = 0.5,
    min_child_weight = 2
    )
clf.fit(X_train,y_train)


def predict_in_dt_high_entropy_in_rf(x):
    neg_entropy = - entropy(rf.predict_proba(x), axis = 1)
    class_anchor = np.square(dt.predict_proba(x)[:,1] - 1).sum()
    return neg_entropy + class_anchor

import langevin

hyps = langevin.Hyperparameters(
    func = predict_in_dt_high_entropy_in_rf,
    beta = 50.,
    eta = 0.05,
    sigma = 0.1,
    reg = lambda x: np.where(np.all((x > 0) & (x < 1), axis =1), 0, np.inf)
    )


np.random.seed(seed = 42)
x_init = np.random.random((X[0:1].shape))
x_last, traj = langevin.adaptive_np_MALA(x_init,hyps, 5000)


fig, ax = plot_wine_features(
            X_train=X_train,
            X_test=X_test, 
            y_train=y_train,
            y_test=y_test,
            generated_samples=traj[-500::10],
            feature_idx1=1,
            feature_idx2=12,
            model= dt,
            feature_names= wine.feature_names
    )


plt.savefig('./images/wineset', dpi=150, bbox_inches='tight')

plt.show()

gen_data = traj[-500::10]
gen_data_X = scaler.inverse_transform(gen_data)
print(f"Prediction probs. DT: {dt.predict_proba(gen_data)}, RF: {rf.predict_proba(gen_data)}")
print(f"Means: {gen_data_X.means(axis = 0)}, standard deviations: {gen_data_X.std(axis = 0)}")