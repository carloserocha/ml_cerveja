# -*- coding:utf-8 -*-
from log import log # def print

import sys
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR

'''
    Imagens de Treinamento
'''
h1 = cv2.imread('imagens/heineken/1.jpg')
h2 = cv2.imread('imagens/heineken/2.jpg')
h3 = cv2.imread('imagens/heineken/3.jpg')
h4 = cv2.imread('imagens/heineken/4.jpg')
s1 = cv2.imread('imagens/skol/1.jpg')
s2 = cv2.imread('imagens/skol/2.jpg')
s3 = cv2.imread('imagens/skol/3.jpg')
s4 = cv2.imread('imagens/skol/4.jpg')

'''
    Imagens de Treinamento em 10x10
'''
h1_10 = cv2.resize(h1, (10,10))
h2_10 = cv2.resize(h2, (10,10))
h3_10 = cv2.resize(h3, (10,10))
h4_10 = cv2.resize(h4, (10,10))
s1_10 = cv2.resize(s1, (10,10))
s2_10 = cv2.resize(s2, (10,10))
s3_10 = cv2.resize(s3, (10,10))
s4_10 = cv2.resize(s4, (10,10))

'''
    Imagem de Teste
'''
test = cv2.imread('imagens/teste.jpg')
test_10 = cv2.resize(test, (10,10))

# Concatena todas as matrizes
X = np.concatenate((h1_10, h2_10, h3_10, h4_10, s1_10, s2_10, s3_10, s4_10), axis=0)


Y = np.array([1,2,3,4,5,6,7,8]).reshape(-1) # Cria um indexador

X = X.reshape(len(Y), -1) # Remodelar X com o tamanho de Y

classifier_linear = SVC(kernel='linear') # Cria o classificar linear

log('Started train of SVC model', True)

classifier_linear.fit(X,Y) # Treina o classificar com imagens e os indexes

log('Finished train', True)

prediction = classifier_linear.predict(test_10.reshape(1,-1)) # Predict the category of image 

score = classifier_linear.score(X,Y) # Score da predição

print('Result: {}'.format(prediction))
print('Score of precision: {:.1f}%'.format(score * 100)) # Mostra o score da predição

# Resulta a imagem da predição
if prediction == 1:
	result = h1
elif prediction == 2:
	result = h2
elif prediction == 3:
	result = h3
elif prediction == 4:
	result = h4
elif prediction == 5:
	result = s1
elif prediction == 6:
	result = s2
elif prediction == 7:
	result = s3
elif prediction == 8:
	result = s4

cv2.imshow("Result", result) # Imagem com base na predição
cv2.imshow("Test", test) # Imagem de Teste
cv2.waitKey(0) # Aguarda a tecla 0

