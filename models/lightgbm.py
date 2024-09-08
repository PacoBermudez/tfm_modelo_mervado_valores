import lightgbm as lgb
from src.config import save_model_params, load_model
import os

def model_lgbmregressor(param_distributions, X, feature_set, nombre_modelo, nuevo_entrenamiento):



    if (not os.path.exists(f'pesos_modelos/{nombre_modelo}_model.pkl')) or (nuevo_entrenamiento == True):

        print("Comienza el entrenamiento del modelo")

        model = lgb.LGBMRegressor(**param_distributions, random_state=42)


        model.fit(
        X[feature_set],
        X["target"]
        )

        save_model_params(model, nombre_modelo)

    else:

        model =load_model(nombre_modelo)

    
    return model