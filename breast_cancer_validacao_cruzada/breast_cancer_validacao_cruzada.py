import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


#importa os dados e suas saidas
previsores =  pd.read_csv('../breast_cancer_dataset/entradas_breast.csv')
classe = pd.read_csv('../breast_cancer_dataset/saidas_breast.csv')


def criarRede():
    classificador = Sequential()    
    classificador.add(Dense(units = 16, activation='relu', kernel_initializer='random_uniform', input_dim = 30))
    #Dropout zera alguns valores de entrada para diminuir a chance de overfiting
    classificador.add(Dropout(0.3))
    classificador.add(Dense(units = 32, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.3))
    classificador.add(Dense(units = 1, activation='sigmoid'))
    otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classificador
    
classificador = KerasClassifier(build_fn=criarRede,epochs=100, batch_size=10)

# resultados por fatia da base de dados
resultado = cross_val_score(estimator=classificador, X = previsores,y = classe, cv = 10, scoring="accuracy")

media = resultado.mean() #media das pontuações
desvio = resultado.std() #desvio
#desvios grandes podem significar overfiting, super treinamento