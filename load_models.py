# -*- coding: utf-8 -*-
# %%
from models.lineal_regresion import model_lineal_regresion as rg
from src.config import load_features, load_balanced_data, load_validation_data
from models.ramdom_forest import model_ramdom_forest as rf
from models.lightgbm import model_lgbmregressor as lgbm
import metric_visualization as metric

# %%
feature_set = load_features("medium")

train = load_balanced_data(feature_set)

validation = load_validation_data(feature_set, train)

# %% [markdown]
# # Regresion lineal

# %%
model = rg(train, feature_set, "Regresion_lineal")

# %%
metric.extract_metrics(validation, model, feature_set, "Regresion_lineal")

# %% [markdown]
# # Ramdom Forest

# %%
param_distributions = {
    'n_estimators': [50,],  # Número de árboles en el bosque
    # 'max_depth': [None, 10, 20, 30, 50],  # Profundidad máxima de los árboles
    # 'min_samples_split': [5],  # Número mínimo de muestras requeridas para dividir un nodo
    # 'min_samples_leaf': [ 4],  # Número mínimo de muestras requeridas en cada hoja del árbol
    #'max_features': ['auto', 'sqrt', 'log2']  # Número máximo de características a considerar en cada división
}

model = rf(param_distributions, train, feature_set, "Random_forest")

metric.extract_metrics(validation, model, feature_set, "Random_forest")

# %% [markdown]
# # LightGBM

# %%
lgbm.model_lgbmregressor(train, feature_set, "LGBM")
