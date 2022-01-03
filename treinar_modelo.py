import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

dados = []
rotulos = []
pasta_base_imagens = "base_letras"

imagens = paths.list_images(pasta_base_imagens)
#print(list(imagens))

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[-2]
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    #PADRONIZAR A IMAGEM EM 20X20
    imagem = resize_to_fit(imagem, 20, 20)

    #ADICIONAR UMA DIMENSÃO PARA O KERAS PODER LER A IMAGEM
    imagem = np.expand_dims(imagem, axis=2)

    #ADICIONAR AS LISTAS DE DADOS E RÓTULOS
    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype="float") / 255
rotulos = np.array(rotulos)

#SEPARACAO EM DADOS DE TREINO (75%) E DADOS DE TESTE (25%)
(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

#OneHot Encoding
#CONVERTER COM ONE-HOT ENCODING
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

#SALVAR O LABELBINARIZER EM UM ARQUIVO COM PICKLE
with open("rotulo_modelo.dat", 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

#CRIAR E TREINAR A INTELIGÊNCIA ARTIFICIAL
modelo = Sequential()

#CRIAR AS CAMADAS DA REDE NEURAL
modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#CRIAR A SEGUNDA CAMADA
modelo.add(Conv2D(50, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#CRIAR TERCEIRA CAMADA
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))

#CAMADA DE SAÍDA
modelo.add(Dense(26, activation="softmax"))

#COMPILAR TODAS AS CAMADAS
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#TREINAR IA
modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=100, verbose=1)

#SALVAR O MODELO EM UM ARQUIVO
modelo.save("modelo_treinado.hdf5")
