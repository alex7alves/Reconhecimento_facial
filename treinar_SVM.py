# -*- coding: utf-8 -*-
"""
@author: Alex Alves
Programa para reconhecer pessoas
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
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


def Gravar_Dados(path,objeto):
    f = open(path, "wb")
    f.write(pickle.dumps(objeto))
    f.close()


entrada_treinamento, saida_treinamento = Separar_dados('dataset/treinamento.csv')
entrada_teste, saida_teste = Separar_dados('dataset/teste.csv')

entrada_treinamento = normalize(entrada_treinamento)
entrada_teste = normalize(entrada_teste)

# Codificar valores para inteiros
encoder = LabelEncoder()
saida_treinamento = encoder.fit_transform(saida_treinamento)
saida_teste = encoder.fit_transform(saida_teste)

reconhecer_pessoa = SVC(C=1.0, kernel="linear", probability=True)
reconhecer_pessoa.fit(entrada_treinamento, saida_treinamento )
# Resultados para o dataset de teste
resultados = reconhecer_pessoa.predict(entrada_teste)

matriz_confusao = confusion_matrix(saida_teste, resultados)
report = classification_report(saida_teste, resultados)


# Salvar o model
Gravar_Dados('pesos/SVM/reconhecer.pickle',reconhecer_pessoa)

# Gravar o label encoder (conversor para inteiro)
Gravar_Dados('pesos/SVM/encoder.pickle',encoder)
