import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_breast.json','r')
estrutura_rede = arquivo.read()
arquivo.close()

#impota a estrutura da rede
classificador = model_from_json(estrutura_rede)
#importa os pesos da rede
classificador.load_weights('classificador_braest.h5')

#cria um novo registro para teste
novoRegisto=np.array([[15.80,8.34,118,900,0.10,0.26,0.08,0.134,0.178,0.20,0.05,1098,0.87,
                       4500,145.2,0.005,0.04,0.05,0.015,0.03,0.007,23.15,16.64,178.5,2018,0.14,
                       0.185,0.84,158,0.363]])
#utiliza a rede para uma nova previsão
previsao = classificador.predict(novoRegisto)