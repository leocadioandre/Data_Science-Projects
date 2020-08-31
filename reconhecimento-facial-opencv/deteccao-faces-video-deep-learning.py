import cv2
from mtcnn.mtcnn import MTCNN #detecção de faces utilizando o deep learning
from cv2 import rectangle,circle
from cv2 import imshow,waitKey,destroyAllWindows,VideoCapture

 
# Captura o video da camera
cap = VideoCapture(0)#No notebook seria a porta que indica a webcam
 
# verifique se a captura esta aberta.
if (cap.isOpened() == False): 
  print("Nao foi possivel ler os dados da camera")

# instanciando o detector
face_detector = MTCNN()

# inicializa o loop para a leitura de frames
while(True):
  ret, frame = cap.read() #captura em video dos frames e do ret=captura com sucesso.
  
  # se a leitura tiver sendo feita
  if ret == True:
     
    # executa o detector de faces
    faces = face_detector.detect_faces(frame)
    if(len(faces)>0):
        for face in faces:
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height
            # desenhando o retangulo no frame
            rectangle(frame, (x, y), (x2, y2), (0,0,255), 3)


    # Exibe o frame processado    
    imshow('deteccao faces',frame)
 
    # se pressionar q para a captura e libera a camera
    if waitKey(1) & 0xFF == ord('q'):
      break
 
  else:
    break 
 
# libera o video
cap.release()
 
# fecha a janela da imagem
destroyAllWindows() 