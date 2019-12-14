# Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys
import joblib

def main(dataset_path):
    dataset = pd.read_csv(dataset_path)

    X, Y = getXAndYValues(dataset)

    X = encodeCategoricalData(X)

    # Avoid the Dummy Variable Trap
    X = X[:, :-1]

    significant_variables = determineSignificantVariables(X, Y)

    # fit the model using significant variables
    model = sm.OLS(Y, significant_variables).fit()

    saveModelToFile(model, path = '../REST-API/model.sav')

    # dataset = pd.read_csv(dataset_path)
    # new = dataset.iloc[:, :-1].values
    # new[1,0] = 165349
    # onehotencoder = make_column_transformer((StandardScaler(), [0, 1, 2]), (OneHotEncoder(), [3]))
    # new = onehotencoder.fit_transform(new)

    # # model input is R & D Spend
    # # Predicting the Test set results
    # y_pred = model.predict([[1, new[1,0]]])

    # print(y_pred)

def getXAndYValues(dataset):
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 4].values
    return X, Y

def encodeCategoricalData(X):
    onehotencoder = make_column_transformer((StandardScaler(), [0, 1, 2]), (OneHotEncoder(), [3]))
    X = onehotencoder.fit_transform(X)
    return X

def determineSignificantVariables(X, Y):
    significance_level = 0.05
    X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
    X = X[:, [0, 1, 2, 3, 4, 5]]
    significant_variables = performBackwardsElimination(X, Y, significance_level)
    return significant_variables

# Building the optimal model using Backward Elimination
def performBackwardsElimination(X, Y, significance_level):
    numberOfVariables = len(X[0])
    for i in range(0, numberOfVariables):
        regressor_OLS = sm.OLS(Y, X).fit()
        highestPValue = max(regressor_OLS.pvalues).astype(float)
        if highestPValue > significance_level:
            for j in range(0, numberOfVariables - i):
                if (regressor_OLS.pvalues[j].astype(float) == highestPValue):
                    X = np.delete(X, j, 1)
    print(regressor_OLS.summary())
    return X

def saveModelToFile(model, path):
    joblib.dump(model, path)

if __name__ == "__main__":
   main(sys.argv[1])
