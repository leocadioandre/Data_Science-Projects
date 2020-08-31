# -*- coding: utf-8 -*-

# Importando as bibliotecas necessárias
import cv2

# Solicita o nome e matricula do usuário.
nome = input('Entre com seu nome: ')
matricula = input('Entre com sua matricula: ') 

# Armazena o nome e matricula no arquivo de labelmap.
file = open("labelmap.csv", "a")#a=append
file.write(matricula+','+nome+"\n") 
file.close()

# Carrega o Detector de Face.
detector= cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Inicializa a captura da camera.
cam = cv2.VideoCapture(0)

# Inicializa o contador de imagens
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    k = cv2.waitKey(1)
    
    # Realiza a detecção de face em cada frame.
    faces = detector.detectMultiScale(frame, 1.3, 5)
    
    # Desenha o retângulo na face detectada utilizando as coordenadas.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    # Exibe a imagem.
    cv2.imshow("Coletor de Imagens - Pressione ENTER para gravar", frame)

    if k%256 == 27: #o número da tecla no teclado = esc-27
        # Se pressionar o ESC
        print("saindo...")
        break
    elif k%256 == 13: # Se pressionar o ENTER
        # Converte a imagem para escala de cinza.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Persiste a imagem em disco utilizando o numero de matricula e o contador.
        cv2.imwrite("dataset/user"+'-'+str(matricula)+'-'+str(img_counter)+".jpg", gray[y:y+h,x:x+w])

        print("Imagem gravada!")
        img_counter += 1

cam.release()

cv2.destroyAllWindows()