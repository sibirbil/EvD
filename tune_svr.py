

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def tune_hyp(X_train, y_train, X_test, y_test):
    # Initialize the SVR model
    svr = SVR()

    # Define the hyperparameter grid
    param_grid = {
        'kernel': ['linear'],  # Kernel type
        'C': [0.1, 1, 10, 100],              # Regularization parameter
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],  # Epsilon for the loss function
    }

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # Metric for evaluation
        cv=5,  # Number of cross-validation folds
        verbose=1,  # Print progress
        n_jobs=-1   # Use all CPU cores
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the corresponding score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Evaluate on the test set
    best_svr = grid_search.best_estimator_
    test_score = best_svr.score(X_test, y_test)
    print("Test Score:", test_score)

    return best_svr, grid_search.best_params_, grid_search.best_score_, test_score
