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
    

novoRegisto=np.array([[15.80,8.34,118,900,0.10,0.26,0.08,0.134,0.178,0.20,0.05,1098,0.87,
                       4500,145.2,0.005,0.04,0.05,0.015,0.03,0.007,23.15,16.64,178.5,2018,0.14,
                       0.185,0.84,158,0.363]])

previsao = classificador.predict(novoRegisto)