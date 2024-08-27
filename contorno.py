import cv2

img = cv2.imread('RECONOCIMIENTO FACIAL\contorno.jpg')
grises = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,umbral = cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY)
contorno, jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contorno, -1, (255, 0, 0), 3)

# Mostramos la imagen:
cv2.imshow('Imagen original', img)

#cv2.imshow('Imagen en grises', grises) 
#cv2.imshow('Imagen con umbral', umbral)

cv2.waitKey(0) 
cv2.destroyAllWindows() # Cerramos todas las ventanas abiertas.
