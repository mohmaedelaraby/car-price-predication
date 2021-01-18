import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#import dataset
dataset = pd.read_csv("CarPrice_training.csv")
dataset.dropna(how='any',inplace=True)
#dataset.drop(['CarName'], axis=1)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,2].values

#Encoding Categorical data

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
transformer = ColumnTransformer([('aspiration', OneHotEncoder(), [0])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.str)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(dataset)
print(Y)