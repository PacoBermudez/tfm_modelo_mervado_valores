# -*- coding: utf-8 -*-
# %%
from models.lineal_regresion import model_lineal_regresion as rg
from src.config import load_features, load_balanced_data, load_validation_data
from models.ramdom_forest import model_ramdom_forest as rf
from models.lightgbm import model_lgbmregressor as lgbm
from models.m_xgboost import model_xgboost as xgb
import metric_visualization as metric

# %%
feature_set = load_features("medium")

train = load_balanced_data(feature_set)

validation = load_validation_data(feature_set, train)

# %%

# %% [markdown]
# # Regresion lineal

# %%
model = rg(train, feature_set, "Regresion_lineal", False)

# %%
metric.extract_metrics(validation, model, feature_set, "Regresion_lineal")

# %% [markdown]
# # Ramdom Forest

# %%
param_distributions = {
    'n_estimators': [10,20,50],  # Número de árboles en el bosque
    'max_depth': [10,20],  # Profundidad máxima de los árboles
    # 'min_samples_split': [5],  # Número mínimo de muestras requeridas para dividir un nodo
    # 'min_samples_leaf': [ 4],  # Número mínimo de muestras requeridas en cada hoja del árbol
    #'max_features': ['auto', 'sqrt', 'log2']  # Número máximo de características a considerar en cada división
}

model = rf(param_distributions, train, feature_set, "Random_forest", False)
print("Extrayendo métricas")
metric.extract_metrics(validation, model, feature_set, "Random_forest")

# %% [markdown]
# # LightGBM

# %%
model = lgbm(train, feature_set, "LGBM", False)
metric.extract_metrics(validation, model, feature_set, "LGBM")

# %% [markdown]
# # XGBoost

# %%


params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 50,  # Reducción de profundidad
    'n_estimators': 50,  # Reducción del número de árboles*
    'tree_method': 'hist',
    'verbosity': 1,
    'colsample_bylevel': 0.7,
    'device' : "cuda"
}  
model = xgb(params, train, feature_set, "XGBoost", True)

# %%
metric.extract_metrics(validation, model, feature_set, "LGBM")

# %%
