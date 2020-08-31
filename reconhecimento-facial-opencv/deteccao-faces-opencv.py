

#Instalacao do OpenCV com Python.
#pip install opencv-python

#Apertar shift + Enter para inicializar o Python

#importando a biblioteca
import cv2

# verificando a versao do OpenCV
print("OpenCV: {}".format(cv2.__version__))

# Importando os métodos necessários
from cv2 import imread
from cv2 import CascadeClassifier # Modelos de ML pré treinado para classificadores, no caso o mesmo é detecção frontal de face.

# carregando a imagem de teste e tranformando em um array numpy
img_array = imread('imagens/woman.jpeg')

# carregando o modelo pre-treinado
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# realizando a detecção da face
#bboxes = classifier.detectMultiScale(img_array)
bboxes = classifier.detectMultiScale(img_array,minNeighbors=8)
#Para diminuir a chance de falsos positivos pode-se especificar o minNeighbors que corresponde ao número de vizinhos. 
#Indica o quanto de janelas atreves da face de detecção, ele considera pra marcar que aquela região é uma face. Ficando mais criterioso.

# Verificando as bounding box
for box in bboxes:
	print("Face encontrada com as coordenadas: {}".format(box))

#plotando um retangulo na face encontrada

#importando o metodo responsável para desenhar o retangulo 
from cv2 import rectangle
from cv2 import imshow,waitKey,destroyAllWindows


#Loop em cada face detectada para "desenhar" o retangulo de acordo com suas coordenadas
for box in bboxes:
	# pegando as coordenadas x e y e altura e largura da caixa
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# desenhando o retangulo na imagem
	rectangle(img_array, (x, y), (x2, y2), (0,0,255), 3)

# exibindo a imagem
imshow('Face Detectada', img_array)

waitKey(0)
# fecha a janela da imagem
destroyAllWindows()