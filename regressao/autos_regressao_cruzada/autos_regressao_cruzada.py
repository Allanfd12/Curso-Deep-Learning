import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('../dataset/autos.csv', encoding='ISO-8859-1')

#preimeiramente nos precisamos filtrar os dados relevantes
#iremos apagar uma parte dos dados, pois não nos é interessante
base = base.drop('dateCrawled', axis= 1)
base = base.drop('dateCreated', axis= 1)
base = base.drop('nrOfPictures', axis= 1)
base = base.drop('postalCode', axis= 1)
base = base.drop('lastSeen', axis= 1)

base['name'].value_counts()

#name deve ser apagada devido a sua alta variabilidade
base = base.drop('name', axis= 1)


base['seller'].value_counts()
# devido a pessima variabilidade em seller, contando apenas com 2 categorias
#sendo uma delas com apenas 3 registros, fica claro que esse campo não te um valor util a agregar
#para a estimativa de preço
base = base.drop('seller', axis= 1)

#mesmo se aplica a offerType
base['offerType'].value_counts()

base = base.drop('offerType', axis= 1)

#filtra inconsistencias de valores muito baixos
i1 = base.loc[base.price <=100]
base = base[base.price>100]

#filtra inconsistencias de valores muito altos
i2 = base.loc[base.price >350000]
base = base[base.price<350000]

#filtra inconsistencias de anos muito altoos
i3 = base.loc[base.yearOfRegistration> 2020]
base = base[base.yearOfRegistration<2020]

i4 = base.loc[base.yearOfRegistration< 1900]
base = base[base.yearOfRegistration>1900]

#considerando a porcentagem de veiculos sem tipo definido
#nos podemos se aproveitar de algumas estrategias para criar esses dados
#nos podemos pegar o tipo de veiculo mais comum e colocar como padrão
f1 = base.loc[pd.isnull(base['vehicleType'])]

base['vehicleType'].value_counts() #limousine
#ou, nos podemos ciar uma classe para informar que esse dado não esta presente   

f2 = base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts()

f3 = base.loc[pd.isnull(base['model'])]
base['model'].value_counts()

f4 = base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts()

f5 = base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts()

#manter os valores nulos
valoresSubstituidos  = {'vehicleType' : 'none','gearbox': 'none','model':'none', 'fuelType':'none','notRepairedDamage':'none' }

#substiuir os valores nulos pelos valores que mais aparecem
valoresSubstituidos2  = {'vehicleType' : 'limousine','gearbox': 'manuell','model':'golf', 'fuelType':'benzin','notRepairedDamage':'nein' }

#substitui os valores nulos pelos valores informado
base = base.fillna(value = valoresSubstituidos)

#seleciona a fatia da base que nos interessa
previsores = base.iloc[:,1:13].values
preco_real =base.iloc[:,0].values

#converte dados 'categoricos' em dados numericos, remove palavras e substitui por numeros
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_previsores = LabelEncoder()
#previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])
#previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
#previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
#previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
#previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
#previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
#previsores[:,10] = labelencoder_previsores.fit_transform(previsores[:,10])

# DEPRECIADO
#onehotencoder = OneHotEncoder(categorical_features =[0,1,2,3,5,8,9,10])

#realiza as duas etapas anteriores juntas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

onehotencoder = ColumnTransformer([("norm1", OneHotEncoder(),[0,1,3,5,8,9,10])], remainder="passthrough")
previsores = onehotencoder.fit_transform(previsores).toarray()

regressor = Sequential()
regressor.add(Dense(units = 320, activation='relu', input_dim = 321))
regressor.add(Dense(units = 320, activation='relu'))
regressor.add(Dense(units = 160, activation='relu'))
regressor.add(Dense(units = 160, activation='relu'))
regressor.add(Dense(units = 80, activation='relu'))
regressor.add(Dense(units = 80, activation='relu'))
regressor.add(Dense(units = 1, activation='linear'))
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, batch_size=300, epochs =5)









