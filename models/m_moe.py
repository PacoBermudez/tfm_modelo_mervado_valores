import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
import joblib


def definir_modelos_y_parametros():
    modelos = {
        'ridge': Ridge(),
        'decision_tree': DecisionTreeRegressor(),
        'random_forest': RandomForestRegressor(),
        'gradient_boosting': GradientBoostingRegressor()
    }
    
    parametros = {
        'ridge': {'alpha': [0.1, 1.0, 10.0]},
        'decision_tree': {'max_depth': [3, 5, 10]},
        'random_forest': {'n_estimators': [50, 100, 200]},
        'gradient_boosting': {'learning_rate': [0.01, 0.1, 0.2]}
    }
    
    return modelos, parametros

# 3. Optimización de Hiperparámetros
def optimizar_modelos(modelos, parametros, X_train, y_train):
    modelos_optimizados = {}
    
    for nombre, modelo in modelos.items():
        grid = GridSearchCV(modelo, parametros[nombre], cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        modelos_optimizados[nombre] = grid.best_estimator_
        print(f"Mejor modelo para {nombre}: {grid.best_estimator_}")
    
    return modelos_optimizados

# 4. Ensamblaje de Modelos
def ensamblar_modelos(modelos):
    # Voting Regressor combina múltiples modelos
    voting_model = VotingRegressor(estimators=list(modelos.items()))
    return voting_model

# 5. Validación Cruzada
def validar_modelo(modelo, X, y):
    scores = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Validación cruzada - RMSE: {rmse_scores.mean()} ± {rmse_scores.std()}")
    return rmse_scores.mean()

# 6. Guardar y Cargar Modelos
def guardar_modelo(modelo, filepath):
    joblib.dump(modelo, filepath)

def cargar_modelo(filepath):
    return joblib.load(filepath)

# Programa Principal
def sistema_experto_prediccion(filepath):
    # Paso 1: Cargar y preprocesar datos
    X, y = cargar_y_preprocesar_datos(filepath)
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Paso 2: Definir modelos y parámetros
    modelos, parametros = definir_modelos_y_parametros()
    
    # Paso 3: Optimizar modelos
    modelos_optimizados = optimizar_modelos(modelos, parametros, X_train, y_train)
    
    # Paso 4: Ensamblar modelos
    modelo_ensamblado = ensamblar_modelos(modelos_optimizados)
    modelo_ensamblado.fit(X_train, y_train)
    
    # Paso 5: Validar modelo ensamblado
    rmse = validar_modelo(modelo_ensamblado, X, y)
    
    # Paso 6: Hacer predicciones y evaluar
    predicciones = modelo_ensamblado.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, predicciones))
    print(f"RMSE en conjunto de prueba: {rmse_test}")
    
    # Paso 7: Guardar el modelo
    guardar_modelo(modelo_ensamblado, "modelo_ensamblado.pkl")