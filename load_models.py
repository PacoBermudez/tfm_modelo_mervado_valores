# %%
from models.lineal_regresion import model_lineal_regresion as rg
from src.config import load_features, load_balanced_data, load_validation_data
from models.ramdom_forest import model_ramdom_forest as rf
import metric_visualization

# %%
feature_set = load_features("medium")

train = load_balanced_data(feature_set)

#validation = load_validation_data(feature_set, train)

# %%
model = rg(train, feature_set, "Regresion_lineal")

# %%
model

# %%
