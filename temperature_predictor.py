# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# importing the dataset
dataset = pd.read_csv('Cricket_chirps.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# scaling the data
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
y = np.array(y).reshape(len(y), 1)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

# applying K Neighbors Regression on data
regressor = KNeighborsRegressor(n_neighbors=6)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(y_test)

# plotting training set to prediction based on training input
plt.scatter(sc_X.inverse_transform(X_train), sc_y.inverse_transform(y_train), color='red', s=1)
plt.scatter(sc_X.inverse_transform(X_train), sc_y.inverse_transform(regressor.predict(X_train)), color='blue', s=1)
plt.xlabel('Cricket Chirps per minute')
plt.ylabel('Temperature in degree Celsius')
plt.title('Training set')
plt.show()

# plotting test set to predicted temperature values
plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_test), color='red', s=1)
plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_pred), color='blue', s=1)
plt.xlabel('Cricket Chirps per minute')
plt.ylabel('Temperature in degree Celsius')
plt.title('Test set')
plt.show()

# printing chirps per minute, actual temperature, predicted temperature respectively
X_test = sc_X.inverse_transform(X_test)
y_test = sc_y.inverse_transform(y_test)
y_pred = sc_y.inverse_transform(y_pred)
for i in range(len(y_pred)):
    print(X_test[i], y_test[i], y_pred[i])