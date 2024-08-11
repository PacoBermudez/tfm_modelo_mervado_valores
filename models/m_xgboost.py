from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from src.config import save_model_params, load_model
import os


def model_xgboost(param_distributions, X, feature_set, nombre_modelo, nuevo_entrenamiento):



    if (not os.path.exists(f'pesos_modelos/{nombre_modelo}_model.pkl')) or (nuevo_entrenamiento == True):

        print("Comienza el entrenamiento del modelo")

        X_t= X[feature_set]
        X_target =X["target"]
 
        
        xgb =XGBRegressor(**param_distributions)

  


        xgb.fit(X_t, X_target)


        save_model_params(xgb, nombre_modelo)

    else:

        xgb =load_model(nombre_modelo)

    
    return xgb