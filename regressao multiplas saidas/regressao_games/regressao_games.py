from sklearn.preprocessing import  OneHotEncoder, MinMaxScaler
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.compose import ColumnTransformer 

base = pd.read_csv('../dataset/games.csv')

# remove campos inuteis para o reinamento

base = base.drop('Other_Sales', axis=1)
base = base.drop('Global_Sales', axis=1)
base = base.drop('Developer', axis=1)

# remove linhas que contem valores NaN
# não é a melhor abordagem, especialmente quando a base tem muitos dados faltantes
base = base.dropna(axis=0)

# outra pessima escolha, invez de normalizar os dados
# vei para teste, até rola, mas não faz isso não kkkk
# remove registro em que a venda foi menor que 1
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogo = base.Name
# remove o nome dos jogos, pois são parametros muito especificos
base = base.drop('Name', axis=1)

previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values


onehotencoder = ColumnTransformer([("norm1", OneHotEncoder(),[0,2,3,8])], remainder="passthrough")
previsores = onehotencoder.fit_transform(previsores).toarray()
#normalização dos dados, converte todos os valores para uma escala
#entre 0 e 1, para evitar interferencias relacionadas a escala
scaler = MinMaxScaler()
scaler.fit(previsores)
previsores = scaler.transform(previsores)

#rede neural
