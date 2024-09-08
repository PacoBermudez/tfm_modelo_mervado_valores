from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from src.config import save_model_params, load_model
import os


def model_ramdom_forest(param_distributions, X, feature_set, nombre_modelo, nuevo_entrenamiento):



    if (not os.path.exists(f'pesos_modelos/{nombre_modelo}_model.pkl')) or (nuevo_entrenamiento == True):

        print("Comienza el entrenamiento del modelo")

        model = RandomForestRegressor(**param_distributions, random_state=42)

        print("Model")
        
        model.fit(
            X[feature_set],
            X["target"]
            )
        


        save_model_params(model, nombre_modelo)

    else:

        model =load_model(nombre_modelo)

    
    return model


