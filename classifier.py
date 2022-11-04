import os
import numpy as np
import cv2

## load images files on dataset-cartas folder, return a numpy array
def loadDataSet():
    path = os.getcwd() + "/dataset-cartas"
    writePath = os.getcwd() + "/dataset-cropped"
    files = os.listdir(path)
    for file in files:
        img = cv2.imread(path + "/" + file)
        crop_img = img[0:100, 0:100]
        #cv2.imwrite(os.path.join(writePath , file),crop_img)
    
    
    
    

def main():
    loadDataSet()
    print("hello world")


if __name__=="__main__":
    main()