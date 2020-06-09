# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('AQI_RND.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("Keys of iris_dataset:\n", dataset.keys())
print("Target names:", dataset['CO Class'])
print("Feature names:\n",dataset[1:3,:])

# # Encoding categorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# print(X)


# Training the Multiple Linear Regression model on the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# X_new = np.array([[1.267,94.98]])
# print("X_new.shape:", X_new.shape)

# prediction = regressor.predict(X_new)
# print("Prediction:", prediction)
# print("Predicted target name:",
#        dataset['CO Class'][prediction])

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


X_new = np.array([[1.267,94.98]])
print("X_new.shape:", X_new.shape)


prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
        dataset['CO Class'][prediction])


y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))