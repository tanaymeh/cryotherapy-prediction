import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

data = pd.read_excel('Desktop/datasets/Cryotherapy.xlsx')

X = data.iloc[:,:6]
Y = data.iloc[:,-1]

X = X.fillna(X.mean())
Y = Y.fillna(Y.mean())

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.20, random_state=0)

sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.fit_transform(test_X)

model = Sequential()

model.add(Dense(units=10,input_dim=6,kernel_initializer='uniform',activation='sigmoid'))
model.add(Dense(units=10,kernel_initializer='uniform',activation='sigmoid'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_X,train_Y,batch_size=10,epochs=100)

predictionY = model.predict(test_X)
predictionY = predictionY>0.7

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y,predictionY)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def BuildModel():
    model = Sequential()
    model.add(Dense(units=10,input_dim=6,kernel_initializer='uniform',activation='sigmoid'))
    model.add(Dense(units=10,kernel_initializer='uniform',activation='sigmoid'))
    model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=BuildModel, batch_size=10, epochs=100)
accuracy = cross_val_score(estimator=model, X=train_X, y=train_Y, cv=10, n_jobs=1)

mean, variance = accuracy.mean(), accuracy.std()



