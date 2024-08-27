import cv2

capturaVideo = cv2.VideoCapture(0)

if not capturaVideo.isOpened():
    print("No se ha podido abrir la cámara")
    exit()
while True:
    tipocamara, camara= capturaVideo.read()
    
    cv2.imshow("Video", camara)
    if cv2.waitKey(1) == ord('q'):
        break
capturaVideo.release()
cv2.destroyAllWindows()