 # -*- coding: utf-8 -*-

'''
    Autor : Alex Alvws.
    
    Programa para detectar face e extrair os descritores delas

'''
import cv2
import os
import numpy as np
import pandas as pd


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
            print(len(faces_detectadas))
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


                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                    descritor.setInput(faceBlob)
                    vec = descritor.forward()
                    c =c + 1
                    print(" tamanho da face", str(c))
                    print(inicioX, inicioY, fimX, fimY)
                    print(vec.flatten())
                    print(type(vec.flatten()))

                    #d = pd.Series({'Descritores': vec.flatten(), 'Pessoa': pessoa_dir})
                    #d = pd.Series({vec.flatten(), 'Pessoa': pessoa_dir})
                    d = list(vec.flatten())
                    d.append(pessoa_dir)
                    lista_descritores.append(d)

                    # descomente essas 3 linhas caso queira ver os rostos detectados
                    #cv2.rectangle(imagem, (inicioX, inicioY), (fimX, fimY), (0, 255, 0), 2)
                    #cv2.imshow("Output", imagem)
                    #cv2.waitKey(0)

    df = pd.DataFrame(lista_descritores)
    df.to_csv(saida)


# Carregar modelos processados
modelo = "res10_300x300_ssd_iter_140000.caffemodel"
config = "deploy.prototxt"
modelDescritor = "openface_nn4.small2.v1.t7"
Extrair_descritores("imagens/treinamento/", config, modelo, modelDescritor, "dataset/treinamento.csv")
Extrair_descritores("imagens/teste/", config, modelo, modelDescritor, "dataset/teste.csv")
