import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


#importa os dados e suas saidas
previsores =  pd.read_csv('../breast_cancer_dataset/entradas_breast.csv')
classe = pd.read_csv('../breast_cancer_dataset/saidas_breast.csv')



classificador = Sequential()    
classificador.add(Dense(units = 16, activation="relu", kernel_initializer="normal", input_dim = 30))
#Dropout zera alguns valores de entrada para diminuir a chance de overfiting
classificador.add(Dropout(0.3))
classificador.add(Dense(units = 16, activation="relu", kernel_initializer="normal"))
classificador.add(Dropout(0.3))
classificador.add(Dense(units = 1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=5,epochs=100)
    
classificador_json = classificador.to_json() #salvaas configurações da rede neural

with open('classificador_breast.json','w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('classificador_braest.h5')