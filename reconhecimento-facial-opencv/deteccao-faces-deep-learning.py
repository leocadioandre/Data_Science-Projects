
#Instalacao do mtcnn (Detecção de faces utilizando machine learning)
#pip install mtcnn

#Necessária a instalação do Tensorflow
#pip install tensorflow

#Verificando a instalação da biblioteca
pip show mtcnn

#importando a biblioteca
import mtcnn

# verificando a versao do mtcnn
print("MTCNN: {}".format(mtcnn.__version__))

# Importando os métodos necessários
from mtcnn.mtcnn import MTCNN #método para fazer detecção de face, usaremos o OpenCV para fazer a manipulação de imagens. Não utilizando o cascate
from cv2 import imread

# carregando a imagem de teste
img_array = imread('imagens/woman.jpeg')

# instanciando o detector
face_detector = MTCNN()

# realizando a detecção da face
faces = face_detector.detect_faces(img_array)

# verificando o tipo do objeto
print(type(faces))

# listando as faces detectadas
for face in faces:
	print(face)

# verificando as bounding boxes detectadas
for face in faces:
	print(face['box'])

# verificando os demais atributos
for face in faces:
	print("Coordenadas bounding box:{}".format(face['box']))
	print("Confianca:{}".format(face['confidence']))
	print("Pontos da Face:{}".format(face['keypoints']))

# plotando um retangulo na face encontrada

# importando o metodo responsável para desenhar o retangulo 
from cv2 import rectangle,circle
from cv2 import imshow,waitKey,destroyAllWindows

for face in faces:
	x, y, width, height = face['box']
	x2, y2 = x + width, y + height
	# desenhando o retangulo na imagem
	rectangle(img_array, (x, y), (x2, y2), (0,0,255), 3)
	# desenhando um ciculo em cada ponto detectado da face
	for ponto,valor in face['keypoints'].items():
		circle(img_array, valor, radius=8, color=(255,0,0))

# exibindo a imagem
imshow('face detectada', img_array)

waitKey(0)
# fecha a janela da imagem
destroyAllWindows()
