# python3 LinearRegression.py
# SIMPLE LINEAR REGRESSION

''' IMPORT LIBRARIES  '''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

''' IMPORT DS '''

url = 'HousePrice.csv'
DS = pd.read_csv( url )

print("\n\n-> DATASET SHAPE\n",DS.shape)
print( '\n\n-> FIRST 5 ROWS\n' )
print(DS.head(5))
print("\n\n")

''' DESCRIPTION OF DATASET '''

DS.describe()

''' EXTRACTING DEPENDENT & INDEPENDENT VARIABLES  '''

X = DS.iloc[:, :-1].values
Y = DS.iloc[:, -1].values

''' SPLITTING the DS INTO TRAINING SET AND TEST SET '''

from sklearn.model_selection import train_test_split
X_Training, X_Test, Y_Training, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print("\n\n-> X TRAIN SHAPE\n",X_Training.shape)
print("\n\n-> Y TRAIN SHAPE\n",Y_Training.shape)

print("\n\n-> X TEST SHAPE\n",X_Test.shape)
print("\n\n-> Y TEST SHAPE\n",Y_Test.shape)

''' LINEAR REGRESSION MODEL : TRAINING'''

from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_Training, Y_Training)

# ''' REGRESSION INTERCEPT & COEFF '''

# print("\n\n-> REGRESSION INTERCEPT : ",Regressor.intercept_)
# print("\n-> REGRESSION COEFF : ",Regressor.coef_)
# print("\n\n")


# ''' PREDICTING TEST SET RESULTS '''

# y_pred = Regressor.predict(X_Test)

# ''' PRINTING ACTUAL VS PREDICTED PRICE '''

# test_data = pd.DataFrame({'Actual': Y_Test, 'Predicted': y_pred})
# print("\n\n-> TEST DATA \n")
# print(test_data)

# ''' ERROR '''

# from sklearn import metrics

# print('\n\n-> MEAN ABSOLUTE ERROR:', metrics.mean_absolute_error(Y_Test, y_pred))
# print('\n-> MEAN SQUARED ERROR:', metrics.mean_squared_error(Y_Test, y_pred))
# print('\n-> ROOT MEAN SQUARED ERROR:', np.sqrt(metrics.mean_squared_error(Y_Test, y_pred)))

# ''' VISUALISING THE TRAINING SET RESULTS '''

# plt.scatter(X_Training, Y_Training, color='red')
# plt.plot(X_Training, Regressor.predict(X_Training), color='blue')
# plt.title('PRICE vs AREA (TRAINING SET)')
# plt.xlabel('Area')
# plt.ylabel('Price')
# plt.show()

# ''' VISUALISING THE TEST SET RESULTS '''

# plt.scatter(X_Test, Y_Test, color='red')
# plt.plot(X_Training, Regressor.predict(X_Training), color='blue')     # train values used here because regression line is made of train set
# plt.title('PRICE vs AREA (TEST SET)')
# plt.xlabel('Area')
# plt.ylabel('Price')
# plt.show()

