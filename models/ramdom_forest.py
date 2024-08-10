from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

def model_ramdom_forest(param_distributions, X, Y, feature_set):
    model = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=1,  # Número de combinaciones aleatorias a probar
        scoring='neg_mean_squared_error',  # Métrica de evaluación
        cv=3,  # Número de pliegues de validación cruzada
        random_state=42,
        n_jobs=-1,  # Utiliza todos los núcleos disponibles
        verbose=2
    )
    random_search.fit(
        X[feature_set],
        X["target"]
        )
    
    results = random_search.cv_results_
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='rank_test_score')

    relevant_columns = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    
    results_df[relevant_columns].to_excel("Resultado_ramdon_forest.xlsx")

    best_model = random_search.best_estimator_


    return best_model


