import cv2
import numpy as np 

valorGauss = 3
valorKernel = 3
original = cv2.imread('RECONOCIMIENTO FACIAL\monedas.jpg')
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gris, (valorGauss, valorGauss), 0)
canny = cv2.Canny(gauss, 60, 100)
kernel = np.ones((valorKernel, valorKernel), np.uint8) # Mostrar en la matriz los numeros enteros de 8 bytes
cerrarMatriz = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

contornos, jerarquia = cv2.findContours(cerrarMatriz.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos, -1, (0, 0, 255), 2)

# Mostrar resultados:

cv2.imshow('Restultado', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
