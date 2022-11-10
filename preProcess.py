"""
Script para fazer o pre processamento do dataset
de cartas usado no trabalho de Inteligência Computacional.
Para usar este script é necessário somente rodar
python preProcess.py, e as imagens serão geradas na pasta
database/dataset_cropped
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

## Função para carregar o dataset de cartas
def loadDataSet():
    path = os.getcwd() + "/database/dataset_cartas"
    writePath = os.getcwd() + "/database/dataset_cropped"
    files = os.listdir(path)
    for file in files:
        print(file)
        try:
            img = cv2.imread(path + "/" + file)
            crop_img = cropImage(img)   
            cv2.imwrite(os.path.join(writePath , file),crop_img)
        except:
            print("error")
    
# Função para cortar a imagem
def cropImage(img):
    # converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # aplica um blur para reduzir o ruido
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    # encontra os contornos
    points = np.argwhere(thresh_gray==0) # encontrar todos os pontos pretos
    points = np.fliplr(points) # salvar os pontos em coordenadas x,y
    x, y, w, h = cv2.boundingRect(points) # encontrar o retangulo que envolve todos os pontos
    x, y, w, h = x-10, y-10, w+20, h+20 # adicionar um pouco de margem
    crop = img[y:y+h, x:x+w] # cortar a imagem
    return crop

def main():
    loadDataSet()

if __name__=="__main__":
    main()