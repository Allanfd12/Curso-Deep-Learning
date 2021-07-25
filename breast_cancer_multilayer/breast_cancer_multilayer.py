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
classificador.add(Dense(units = 16, activation='relu', 
                        kernel_initializer='random_uniform'))
# camada de saida
classificador.add(Dense(units = 1, activation='sigmoid'))

otimizador = keras.optimizers.Adam(lr=0.0001, decay=0.00001, clipvalue=0.5)

classificador.compile(optimizer=otimizador, loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

# cria a rede e define alguns parametros de treinamento
#classificador.compile(optimizer='adam', loss='binary_crossentropy',
#                      metrics=['binary_accuracy'])

# batch_size = numero de registros até atualizar pesos
# epochs = numero de ciclos completos de interação com os dados
classificador.fit(previsores_treinamento,classe_treinamento, batch_size=10, epochs=100)

#lÊ os pesos
pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

#executa a rede em uma base de teste
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes >0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
#calcula a precisão
precisao = accuracy_score(classe_teste,previsoes)
#calcula a matriz de confusão
matrix = confusion_matrix(classe_teste,previsoes)

resultado = classificador.evaluate(previsores_teste,classe_teste)
