import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
# Dataset import section
dataset = pd.read_csv('CarPrice_training.csv')
dataset.dropna(how='any',inplace=True)
#One_hot_encoding


#Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset['symboling']=encoder.fit_transform(dataset['symboling'])
dataset['CarName']=encoder.fit_transform(dataset['CarName'])
dataset['fueltype']=encoder.fit_transform(dataset['fueltype'].astype(str))
dataset['aspiration']=encoder.fit_transform(dataset['aspiration'].astype(str))
dataset['doornumber']=encoder.fit_transform(dataset['doornumber'].astype(str))
dataset['carbody']=encoder.fit_transform(dataset['carbody'].astype(str))
dataset['drivewheel']=encoder.fit_transform(dataset['drivewheel'].astype(str))
dataset['enginelocation']=encoder.fit_transform(dataset['enginelocation'].astype(str))
dataset['cylindernumber']=encoder.fit_transform(dataset['cylindernumber'].astype(str))
dataset['enginetype']=encoder.fit_transform(dataset['enginetype'].astype(str))
dataset['aspiration']=encoder.fit_transform(dataset['aspiration'].astype(str))
dataset['fuelsystem']=encoder.fit_transform(dataset['fuelsystem'].astype(str))



#plt.figure(figsize=(25,20))
#sns.countplot(dataset['fuelsystem'])
#sns.countplot(dataset['enginetype'])
#sns.countplot(dataset['enginelocation'])
#sns.countplot(dataset['doornumber'])
#sns.countplot(dataset['horsepower'])



plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(), annot = True, cmap = "RdYlGn")
plt.show()

x = dataset.iloc[:, 21:25].values
y = dataset['price'].values

#Splitting dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .30, shuffle=True)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print("Multi _reg")
print("Train Accuracy:", regressor.score(x_train, y_train))
print("Test Accuracy:", regressor.score(x_test, y_test))
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), y_pred))
true_Carprice_value=np.asarray(y_test)[0]
predicted_Carprice_value=y_pred[0]

print('True value for the first player in the test set in millions is : ' + str(true_Carprice_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_Carprice_value))
#######################################################################33333333333333333333333
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=4)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(x_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(x_test))

print("poly_reg")
#print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

true_player_value=np.asarray(y_test)[0]
predicted_player_value=prediction[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))

#######################################################################33333333333333333333333
