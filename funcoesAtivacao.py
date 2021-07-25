import numpy as np

def step(soma): # uso em problemas linearmente separaveis
    if (soma >=1):
        return 1
    return 0 

def sigmoid(soma): #problemas de logica binaria
    return 1/(1+np.exp(-soma))

def tangenteHiperbolica(soma): # utilem classificação
    return (np.exp(soma)-np.exp(-soma))/(np.exp(soma)+np.exp(-soma))
            
def relu(soma): # importante para redes neurai covolucionais
    if(soma >=0):
        return soma
    return 0

def linear(soma): # importante para regresaão
    return soma

def softmax(x): # importante para problemas de classificação com multiplas classes de dados
    ex = np.exp(x)
    return ex / ex.sum()

soma = 0.358
testeStep = step(soma)
testeSigmoid = sigmoid(soma)
testeHiperbolica = tangenteHiperbolica(soma)
testeRelu = relu(soma)
testeLinear = linear(soma)

valores = [1.0, 0.2, 5.3,]
print(softmax(valores ))