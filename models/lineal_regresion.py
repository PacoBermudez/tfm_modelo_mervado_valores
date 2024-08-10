from sklearn.linear_model import LinearRegression
from src.config import save_model_params

def model_lineal_regresion(X,feature_set, nombre_modelo):
    model = LinearRegression()
    model.fit(
        X[feature_set],
        X["target"]
        )
    
    save_model_params(model, nombre_modelo)
    return model


