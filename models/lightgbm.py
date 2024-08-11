import lightgbm as lgb
from src.config import save_model_params, load_model
import os

def model_lgbmregressor(X, feature_set, nombre_modelo, nuevo_entrenamiento):



    if (not os.path.exists(f'pesos_modelos/{nombre_modelo}_model.pkl')) or (nuevo_entrenamiento == True):

        print("Comienza el entrenamiento del modelo")

        model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=2**5-1,
        colsample_bytree=0.1
        )

        # This will take a few minutes 🍵
        model.fit(
        X[feature_set],
        X["target"]
        )

        save_model_params(model, nombre_modelo)

    else:

        model =load_model(nombre_modelo)

    
    return model