# -*- coding: utf-8 -*-
"""
@author: Alex Alves

Programa para reconhecer pessoa em uma imagem

"""
import os
import cv2
import numpy as np
from keras.models import model_from_json

img = "testes/teste1.jpeg"
imagem = cv2.imread(img)

# Carregar modelos processados
modelo = "res10_300x300_ssd_iter_140000.caffemodel"
config = "deploy.prototxt"
modelDescritor = "openface_nn4.small2.v1.t7"

# Carregando estrutura da rede
arquivo = open('pesos/reconhecer.json','r')
esrutura_rede = arquivo.read()
arquivo.close()

reconhecer = model_from_json(esrutura_rede)
# Carregando os pesos
reconhecer.load_weights('pesos/reconhecer.h5')

 # rede neural para detectar faces
detector = cv2.dnn.readNetFromCaffe(config, modelo)
# Rede neural para extrair os descritores
descritor = cv2.dnn.readNetFromTorch(modelDescritor)

(h, w) = imagem.shape[:2]

# Construindo o blob para a imagem -> a rede tem entrada de 300 x 300
imageBlob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

# aplicando deep-learning para detectar faces em imagens
detector.setInput(imageBlob)
faces_detectadas = detector.forward()

# pega os nomes das pessoas dos diretorios
pessoas = os.listdir("imagens/treinamento")
 
# ordenando a lista - esta vindo desordenada
pessoas = sorted(pessoas)

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
        
        # Rodar rede treinada para reconhecimento
        resultado = reconhecer.predict(vec)
        saida_rede = np.argmax(resultado)
        
        # Escrevendo nome da pessoa identificada na imagem
        cv2.putText(imagem,pessoas[saida_rede], (inicioX -10, inicioY -10),
		                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Mostrando caixa delimitadora ao redor do rosto detectado
        cv2.rectangle(imagem, (inicioX, inicioY), (fimX, fimY), (0, 255, 0), 2)
        # Mostrando imagem para o usuario
        cv2.imshow("Output", imagem)
        cv2.waitKey(0)






