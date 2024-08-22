import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def data_processing(train):
    df_processing = train.groupby("era")["target"].mean().reset_index()

    return df_processing

    


def model_sarimax(train, feature_set):

    X_t= train[feature_set]
    X_target =train["target"]
    
    model = SARIMAX(X_target, 
                exog=X_t,
                order=(0, 2, 2), 
                seasonal_order=(0, 2, 2, 5),
                enforce_stationarity=False, 
                enforce_invertibility=False)

    # Entrenar el modelo
    model = model.fit(disp=False)

    return model