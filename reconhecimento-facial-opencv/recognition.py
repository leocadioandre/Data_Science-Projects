# -*- coding: utf-8 -*-

# Importando as bibliotecas necessárias.
import cv2
import numpy as np
import csv

# Instanciando o algoritmo LBPH.
recognizer = cv2.face.LBPHFaceRecognizer_create(1,10,8,8)

# Carregando o arquivo do modelo em disco.
recognizer.read('model/model_lbph.yml')

# Carregando o Classificador Cascade para detecção da face.
cascadePath = "model/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Inicializa a captura de camera.
cam = cv2.VideoCapture(0)

# Definição de fonte, tamanho e cor.
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)

# Lendo o arquivo labelmap para pegar o codigo e nome das classes.
with open('labelmap.csv', mode='r') as infile:
    reader = csv.reader(infile)
    mydict = {rows[0]:rows[1] for rows in reader}

label = ''

while True:
    # Inicializa a leitura da camera.
    ret, frame = cam.read()
    
    # Converte a imagem em escala de cinza.
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Detecta as faces. 
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        
        # Desenha o retângulo ao redor da face detectada utilizando as coordenadas.
        cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
        
        # Faz a predição para a face detectada.
        mat, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        # Converte a matricula para string.
        mat = str(mat)
        
        # Verifica a confiança retornada pelo algoritmo.
        if(conf > 30):
            # Recupera o nome da classe no dicionário.
            label = mydict.get(mat)
            conf = "  {0}%".format(round(conf))
            
            # Imprime a classe e confiança no console.
            print (label,conf)

        else:
            # Define a classe como Desconhecido caso o valor esteja abaixo do grau de confiança.
            label="Desconhecido"
            conf = "  {0}%".format(round(conf))
            # Imprime a classe e confiança no console.
            print (label,conf)
        
        # Imprime o nome da classe e confiança na tela.
        cv2.putText(frame,str(label), (x,y+h), fontFace, fontScale, fontColor)
        cv2.putText(frame,str(conf), (x+30,y+h+30), fontFace, fontScale, fontColor) 

    # Exibe o frame com os dados da predição.
    cv2.imshow('Reconhecimento Facial',frame) 
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
