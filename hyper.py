from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import get_boston_housing

# Load dataset
data = get_boston_housing()
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0],
        "solver": ["auto", "svd", "cholesky"],
        "fit_intercept": [True, False]
    },
    "Decision Tree": {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["squared_error", "friedman_mse"]
    },
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }
}

models = {
    "Ridge": Ridge(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

for model_name in models:
    print(f"\nTuning {model_name}...")
    search = GridSearchCV(models[model_name], param_grid[model_name], cv=5, scoring='r2')
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    print(f"Best Params for {model_name}: {search.best_params_}")
    print(f"{model_name} - MSE: {mean_squared_error(y_test, predictions):.2f}, R²: {r2_score(y_test, predictions):.2f}")