# -*- coding: utf-8 -*-

# Importando as bibliotecas necessárias.
import cv2, os
import numpy as np
from PIL import Image

# Instanciando o algoritmo LBPH.
# parametros: radius, neighbors, grid_x e grid_y
recognizer = cv2.face.LBPHFaceRecognizer_create(1,10,8,8)

# Carrega o Detector de Face.
detector= cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

# Coleta todas as imagens e seus ids.
def coleta_imagens_matriculas(path):
    
    # Lista o caminho completo de todas as imagens do diretório.
    imagens = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Lista de faces.
    lista_faces = []
    
    # Lista de matriculas.
    matriculas = []
    
    # Carregando cada uma das imagens no disco.
    for imagem in imagens:
        
        # Carregando a imagem.
        obj_image = Image.open(imagem)
        
        # Convertendo a imagem carregada em array numpy.
        imagem_array = np.array(obj_image,'uint8')#imagem 8 pixels
        
        # Pegando o valor de matricula atribuido.    
        matricula = int(os.path.split(imagem)[-1].split("-")[1])
        
        # Incluindo imagens e matriculas correspondentes nas listas.
        lista_faces.append(imagem_array)

        matriculas.append(matricula)

    return lista_faces, matriculas


faces, matriculas = coleta_imagens_matriculas('dataset')

# Treinando o algoritmo com as imagens das faces e suas respectivas matriculas.
recognizer.train(faces, np.array(matriculas))

# Salva o modelo em disco.
recognizer.save('model/model_lbph.yml')