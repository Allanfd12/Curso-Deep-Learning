import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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

#from sklearn.model_selection import train_test_split
#previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation='relu', input_dim=4))
   # classificador.add(Dense(units = 4, activation='relu'))
    classificador.add(Dense(units = 3, activation='softmax'))
    classificador.compile(optimizer='adam',loss='categorical_crossentropy', metrics =['categorical_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn= criar_rede, epochs = 1000, batch_size = 1)

resultados = cross_val_score(estimator=classificador, X= previsores, y= classe, cv = 5, scoring= "accuracy")

media = resultados.mean()
desvio = resultados.std()