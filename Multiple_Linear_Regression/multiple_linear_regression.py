# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

onehotencoder = make_column_transformer((StandardScaler(), [0, 1, 2]), (OneHotEncoder(), [3]))
X = onehotencoder.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, :-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
def backwardsElimination(data, significance_level):
    numberOfVariables = len(data[0])
    for i in range(0, numberOfVariables):
        regressor_OLS = sm.OLS(y, data).fit()
        highestPValue = max(regressor_OLS.pvalues).astype(float)
        if highestPValue > significance_level:
            for j in range(0, numberOfVariables - i):
                if (regressor_OLS.pvalues[j].astype(float) == highestPValue):
                    data = np.delete(data, j, 1)
    print(regressor_OLS.summary())
    return data

significance_level = 0.05
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
data = X[:, [0, 1, 2, 3, 4, 5]]
model = backwardsElimination(data, significance_level)
