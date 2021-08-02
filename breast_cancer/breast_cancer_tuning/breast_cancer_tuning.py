import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


#importa os dados e suas saidas
previsores =  pd.read_csv('../breast_cancer_dataset/entradas_breast.csv')
classe = pd.read_csv('../breast_cancer_dataset/saidas_breast.csv')


def criarRede(optimizer,loss, kernel_initializer,activation, neurons,dropout):
    classificador = Sequential()    
    classificador.add(Dense(units = 16, activation=activation, kernel_initializer=kernel_initializer, input_dim = 30))
    #Dropout zera alguns valores de entrada para diminuir a chance de overfiting
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 16, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 16, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 12, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 12, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(dropout))

    classificador.add(Dense(units = 1, activation='sigmoid'))
    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return classificador
    
classificador = KerasClassifier(build_fn=criarRede)
#best 10, 100,16,16,8,0
parametros = {'batch_size':[1,5,10],
              'epochs':[110],
              'optimizer':['adam'],
              'loss':['binary_crossentropy'],
              'kernel_initializer':['normal'],
              'activation':['relu'],
              'neurons':[16],
              'dropout':[0.3]}



grid_search = GridSearchCV(estimator=classificador, param_grid=parametros,scoring='accuracy',cv=5)

grid_search = grid_search.fit(previsores,classe)

melhores_parametros = grid_search.best_params_

melhor_precisao = grid_search.best_score_

