import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('dataset iris/iris.csv')
previsores = base.iloc[:,0:4].values #dados
classe = base.iloc[:,4].values #resposta

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units = 4, activation='relu', input_dim=4))
classificador.add(Dense(units = 4, activation='relu'))
classificador.add(Dense(units = 3, activation='softmax'))
classificador.compile(optimizer='adam',loss='categorical_crossentropy', metrics =['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)