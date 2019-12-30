# -*- coding: utf-8 -*-

"""
@authores: Alex Alves & Vinicius Brito Filho
Programa para reconhecer pessoas
"""

import cv2
import os
import numpy as np
import pandas as pd
import os.path

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.preprocessing import normalize
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# Função para capturar nomes de arquivos
def Retornar_imagens(caminho):
    for _, _, arquivo in os.walk(caminho):
        pass
    return arquivo


def Abrir_imagem(local, img):
    imagem = cv2.imread(local+"/"+img)
    return imagem


def Carregar_imagens(caminho):
    lista_imagens = []
    arquivos = Retornar_imagens(caminho)
    for x in arquivos:
        img = Abrir_imagem(caminho, x)
        lista_imagens.append(img)
    return lista_imagens

def Extrair_descritores(diretorio, config, modelo, modelDescritor, saida):
    c = 0
    # rede neural para detectar faces
    detector = cv2.dnn.readNetFromCaffe(config, modelo)
    # Rede neural para extrair os descritores
    descritor = cv2.dnn.readNetFromTorch(modelDescritor)
    lista_descritores = []
    diretorios = os.listdir(diretorio)
    for pessoa_dir in diretorios:
        local = diretorio + pessoa_dir

        lista_imagens = Carregar_imagens(local)
        for imagem in lista_imagens:

            (h, w) = imagem.shape[:2]

            # Construindo o blob para a imagem -> a rede tem entrada de 300 x 300
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300),
                        (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # aplicando deep-learning para detectar faces em imagens
            detector.setInput(imageBlob)
            faces_detectadas = detector.forward()
            # Garantir se pelo menos uma face foi encontrada
            if len(faces_detectadas) > 0:
                # Pegar a maior probabilidade
                i = np.argmax(faces_detectadas[0, 0, :, 2])
                confidence = faces_detectadas[0, 0, i, 2]

                #detecções com maiores probabilidades
                if confidence > 0.5:
                    # capturar as cordenadas (x,y) para a caixa delimitadora da face
                    caixa_delimitadora = faces_detectadas[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (inicioX, inicioY, fimX, fimY) = caixa_delimitadora.astype("int")

                    face = imagem[inicioY:fimY, inicioX:fimX]
                    (fH, fW) = face.shape[:2]

                    if (fH > 0 and fW > 0):
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                        descritor.setInput(faceBlob)
                        vec = descritor.forward()
                        c =c + 1
                        print("Extraindo descritores da face: ", str(c))

                        d = list(vec.flatten())
                        d.append(pessoa_dir)
                        lista_descritores.append(d)

    df = pd.DataFrame(lista_descritores)
    df.to_csv(saida)

def Separar_dados(local):
    dataset = pd.read_csv(local, sep=',')

    colunas = []
    for c in range(128):
       colunas.append(str(c))

    descritores = dataset.loc[:,colunas]
    #print(dataset.loc[1:5,col])
    nomes = dataset.loc[:,'128']
    return [descritores, nomes]


# Carregar modelos processados
modelo = "res10_300x300_ssd_iter_140000.caffemodel"
config = "deploy.prototxt"
modelDescritor = "openface_nn4.small2.v1.t7"

for datasetnum in range(1,6):
    if (not(os.path.isfile("dataset/dataset"+str(datasetnum)+".csv"))):
        Extrair_descritores("imagens/dataset"+str(datasetnum)+"/", config, modelo, modelDescritor, "dataset/dataset"+str(datasetnum)+".csv")

    tipos = ["70_30", "60_40", "50_50", "10-fold cv"]
    for tipo in tipos:
        entrada, saida = Separar_dados("dataset/dataset"+str(datasetnum)+".csv")

        entrada = normalize(entrada)

        # # Codificar valores para inteiros
        encoder = LabelEncoder()
        saida = encoder.fit_transform(saida)

        if (tipo == "70_30"):
            entrada_treinamento, entrada_teste, saida_treinamento, saida_teste = train_test_split(entrada, saida, test_size=0.3, random_state=0, stratify=saida)
        elif (tipo == "60_40"):
            entrada_treinamento, entrada_teste, saida_treinamento, saida_teste = train_test_split(entrada, saida, test_size=0.4, random_state=0, stratify=saida)
        elif (tipo == "50_50"):
            entrada_treinamento, entrada_teste, saida_treinamento, saida_teste = train_test_split(entrada, saida, test_size=0.5, random_state=0, stratify=saida)


        print("Dataset: "+str(datasetnum)+" Tipo: "+tipo)
        if (tipo != "10-fold cv"):
            # Converter numeros inteiros para variaveis dummy (one hot encoded)
            dummy_treimaento = np_utils.to_categorical(saida_treinamento)
            dummy_teste = np_utils.to_categorical(saida_teste)

            # Criando a rede neural
            reconhecer_pessoa = Sequential()
            #Adicionando camada de entrada
            reconhecer_pessoa.add(Dense(units=64,activation='sigmoid',kernel_initializer='random_uniform',input_dim=128))

            #Adicionando uma camada oculta
            reconhecer_pessoa.add(Dense(units=32,activation='tanh',kernel_initializer='random_uniform'))
            # Adicionando camada de saida
            reconhecer_pessoa.add(Dense(units=2,activation='softmax'))

            reconhecer_pessoa.summary()
            # Compilar a rede
            otimizar = keras.optimizers.Adam(lr=0.001,decay=0.0001)

            reconhecer_pessoa.compile(otimizar,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

            #  Fazer o treinamento da rede - erro calculado para 10 amostras depois atualiza os pesos -descida do gradiente estocasticos de 1 em 1 amostra
            reconhecer_pessoa.fit(entrada_treinamento, dummy_treimaento, batch_size=1, epochs=150, verbose=0)

            resultado = reconhecer_pessoa.evaluate(entrada_teste, dummy_teste, verbose=0)
            previsoes = reconhecer_pessoa.predict(entrada_teste)
            previsoes_bool = (previsoes > 0.5)

            previsoes_classe = [np.argmax(v) for v in previsoes_bool]
            saida_classe = [np.argmax(v) for v in dummy_teste]

            matriz_confusao = confusion_matrix(previsoes_classe, saida_classe)
            report = classification_report(previsoes_classe, saida_classe, digits=6)
            print("Matriz Confusão:")
            print(matriz_confusao)
            print("Relatório:")
            print(report)
        else:
            skf = StratifiedKFold(n_splits=10)
            contador = 0
            for train, test in skf.split(entrada, saida):
                # Converter numeros inteiros para variaveis dummy (one hot encoded)
                dummy_treimaento = np_utils.to_categorical(saida[train])
                dummy_teste = np_utils.to_categorical(saida[test])

                # create model
                reconhecer_pessoa = Sequential()
                #Adicionando camada de entrada
                reconhecer_pessoa.add(Dense(units=64,activation='sigmoid',kernel_initializer='random_uniform',input_dim=128))

                #Adicionando uma camada oculta
                reconhecer_pessoa.add(Dense(units=32,activation='tanh',kernel_initializer='random_uniform'))
                # Adicionando camada de saida
                reconhecer_pessoa.add(Dense(units=2,activation='softmax'))

                reconhecer_pessoa.summary()
                # Compilar a rede
                otimizar = keras.optimizers.Adam(lr=0.001,decay=0.0001)

                reconhecer_pessoa.compile(otimizar,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

                # Fit the model
                reconhecer_pessoa.fit(entrada[train], dummy_treimaento, batch_size=1, epochs=150, verbose=0)
                # evaluate the model
                resultado = reconhecer_pessoa.evaluate(entrada[test], dummy_teste, verbose=0)
                previsoes = reconhecer_pessoa.predict(entrada[test])
                previsoes_bool = (previsoes > 0.5)

                previsoes_classe = [np.argmax(v) for v in previsoes_bool]
                saida_classe = [np.argmax(v) for v in dummy_teste]

                matriz_confusao = confusion_matrix(previsoes_classe, saida_classe)
                report = classification_report(previsoes_classe, saida_classe, digits=6)
                print("Matriz Confusão "+str(contador)+":")
                print(matriz_confusao)
                print("Relatório "+str(contador)+":")
                print(report)
                contador = contador + 1

        # salvar estrutura da rede e pesos
        reconhecer_json = reconhecer_pessoa.to_json()
        with open('pesos/reconhecer_dataset'+str(datasetnum)+'_'+tipo+'.json', 'w') as json_file:
            json_file.write(reconhecer_json)
        reconhecer_pessoa.save_weights('pesos/reconhecer_dataset'+str(datasetnum)+'_'+tipo+'.h5')
