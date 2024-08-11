from sklearn.linear_model import LinearRegression
from src.config import save_model_params, load_model
import os

def model_lineal_regresion(X,feature_set, nombre_modelo, nuevo_entrenamiento):



    if (not os.path.exists(f'pesos_modelos/{nombre_modelo}_model.pkl')) or (nuevo_entrenamiento == True):

        print("Comienza el entrenamiento del modelo")

        model = LinearRegression()
        model.fit(
            X[feature_set],
            X["target"]
            )
        
        save_model_params(model, nombre_modelo)

        print(f"Finalizado el entrenamiento del modelo {nombre_modelo}")

    else:

        model =load_model(nombre_modelo)

    
    return model


