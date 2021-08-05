import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

#importa o DB
base = pd.read_csv('../dataset iris/iris.csv')
# separa as respostas e os dados em duas variaveis diferentes
previsores = base.iloc[:,0:4].values #dados
classe = base.iloc[:,4].values #resposta

#transforma os dados de valores "Iris-setosa", "Iris-versicolor", Iris-virginica"
#para 
# "Iris-setosa" = 100
# "Iris-versicolor" = 010
# "Iris-virginica" = 001
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

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=2000)

#calcula a precisão com base nos dados de teste
resultado = classificador.evaluate(previsores_teste, classe_teste)

#calcula a previsão manulamente
previsoes = classificador.predict(previsores_teste)

#transforma em booleanos
previsoes = (previsoes > 0.5)

#ajusta o formato da classe de testes, retorna o indice com o maior valor
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]

#ajusta o formato da classe de previsoes retornando o indice de maior valor
previsoes2 = [np.argmax(t) for t in previsoes]


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(previsoes2, classe_teste2)










