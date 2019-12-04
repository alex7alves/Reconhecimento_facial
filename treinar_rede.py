# -*- coding: utf-8 -*-
"""
@author: Alex Alves
Programa para reconhecer pessoas
"""
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize


def Separar_dados(local):
    dataset = pd.read_csv(local, sep=',')

    colunas = []
    for c in range(128):
       colunas.append(str(c))

    descritores = dataset.loc[:,colunas]
    #print(dataset.loc[1:5,col])
    nomes = dataset.loc[:,'128']
    return [descritores, nomes]



entrada_treinamento, saida_treinamento = Separar_dados('dataset/treinamento.csv')
entrada_teste, saida_teste = Separar_dados('dataset/teste.csv')


entrada_treinamento = normalize(entrada_treinamento)
entrada_teste = normalize(entrada_teste)


# encode class values as integers
encoder = LabelEncoder()
saida_treinamento = encoder.fit_transform(saida_treinamento)
saida_teste = encoder.fit_transform(saida_teste)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_treimaento = np_utils.to_categorical(saida_treinamento)
dummy_teste = np_utils.to_categorical(saida_teste)


# Criando a rede neural
reconhecer_pessoa = Sequential()
#Adicionando camada de entrada
reconhecer_pessoa.add(Dense(units=64,activation='sigmoid',kernel_initializer='random_uniform',input_dim=128))

#Adicionando uma camada oculta
reconhecer_pessoa.add(Dense(units=32,activation='sigmoid',kernel_initializer='random_uniform'))
#reconhecer_pessoa.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform'))
# Adicionando camada de saida
reconhecer_pessoa.add(Dense(units=3,activation='sigmoid'))


reconhecer_pessoa.summary()
# Compilar a rede
#compile(descida_gradiente,função do erro- MSE, precisão da rede)

# clipvalue -> delimita os valores dos pesos entre 0.5 e -0.5
# lr = tamanho do passo, decay-> redução do passo
otimizar = keras.optimizers.Adam(lr=0.001,decay=0.0001)
#otimizar = keras.optimizers.Adam(lr=0.01,decay=0.001)
# Nesse caso o clipvalue prejudicou
#otimizar = keras.optimizers.Adam(lr=0.004,decay=0.0001,clipvalue=0.5)

reconhecer_pessoa.compile(otimizar,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#  Fazer o treinamento da rede - erro calculado para 10 amostras
#depois atualiza os pesos -descida do gradiente estocasticos de 1 em 1 amostra
reconhecer_pessoa.fit(entrada_treinamento,dummy_treimaento,batch_size=1,epochs=2000)

resultado = reconhecer_pessoa.evaluate(entrada_teste, dummy_teste)
previsoes = reconhecer_pessoa.predict(entrada_teste)
previsoes_bool = (previsoes > 0.5)

previsoes_classe = [np.argmax(v) for v in previsoes_bool]
saida_classe = [np.argmax(v) for v in dummy_teste]

matriz_confusao = confusion_matrix(previsoes_classe, saida_classe)


# salvar estrutura da rede e pesos
reconhecer_json = reconhecer_pessoa.to_json()
with open('pesos/reconhecer.json', 'w') as json_file:
    json_file.write(reconhecer_json)
reconhecer_pessoa.save_weights('pesos/reconhecer.h5')
