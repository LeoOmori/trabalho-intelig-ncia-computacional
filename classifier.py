import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

## load images files on dataset-cartas folder, return a numpy array
def loadDataSet():
    path = os.getcwd() + "/dataset_cartas"
    writePath = os.getcwd() + "/dataset_cropped"
    files = os.listdir(path)
    for file in files:
        print(file)
        img = cv2.imread(path + "/" + file)
        crop_img = cropImage(img)
        ## wait 5 miliseconds wuthot timer lib
        time.sleep(0.5)
        cv2.imwrite(os.path.join(writePath , file),crop_img)
    
def cropImage(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold to get just the signature
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    # find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray==0) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
    crop = img[y:y+h, x:x+w] # create a cropped region of the gray image
    return crop

def main():
    loadDataSet()
    print("hello world")


if __name__=="__main__":
    main()