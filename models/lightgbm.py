import lightgbm as lgb


def model_lgbmregressor(X,feature_set):

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

    return model 