from sklearn.linear_model import LinearRegression

def model_lineal_regresion(X,Y):
    model = LinearRegression()
    model.fit(
        X,
        Y
        )
    return model


