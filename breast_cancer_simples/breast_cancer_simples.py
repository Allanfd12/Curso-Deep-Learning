import pandas as pd

#importa os dados e suas saidas
previsores =  pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size =0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()
# formula de partida para o units
# entradas + saidas /2
# (30+1)/2 = 15,5 => 16 
# primeira camada oculta
classificador.add(Dense(units = 16, activation='relu', 
                        kernel_initializer='random_uniform', input_dim = 30))
# camada de saida
classificador.add(Dense(units = 1, activation='sigmoid'))

# cria a rede e define alguns parametros de treinamento
classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

# batch_size = numero de registros até atualizar pesos
# epochs = numero de ciclos completos de interação com os dados
classificador.fit(previsores_treinamento,classe_treinamento, batch_size=10, epochs=100)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes >0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste,previsoes)

matrix = confusion_matrix(classe_teste,previsoes)

resultado = classificador.evaluate(previsores_teste,classe_teste)
