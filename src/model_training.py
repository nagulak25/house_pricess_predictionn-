from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import pickle

from src.data_preprocessing import load_data, get_features_target, create_preprocessor
from src.model_evaluation import evaluate_model

def train_models():
    df = load_data('data/house_prices.csv')
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        results[name] = (pipeline, metrics)

    # Select best model (highest R2)
    best_model = max(results.items(), key=lambda x: x[1][1]['r2'])

    print(f"Best Model: {best_model[0]}")

    # Save model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model[1][0], f)

    return results

if __name__ == "__main__":
    results = train_models()
    
    for model_name, (_, metrics) in results.items():
        print(f"\n{model_name}")
        print(f"MAE: {metrics['mae']}")
        print(f"R2: {metrics['r2']}")
        print(f"RMSE: {metrics['rmse']}")