from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import cv2  
import numpy as np
from skimage.feature import SIFT
from skimage import transform
from skimage.feature import hog
from sklearn.decomposition import PCA
from skimage.transform import resize
import os
from sklearn.preprocessing import StandardScaler

class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints + 2))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist, lbp

def lbpExtraction(img):
    desc = LocalBinaryPatterns(3, 8)
    hist, lbp = desc.describe(img)  
    return lbp.flatten()

def glcmExtraction(img):
    graycom = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    # Find the GLCM properties
    contrast = graycoprops(graycom, 'contrast')
    dissimilarity = graycoprops(graycom, 'dissimilarity')
    homogeneity = graycoprops(graycom, 'homogeneity')
    energy = graycoprops(graycom, 'energy')
    correlation = graycoprops(graycom, 'correlation')
    ASM = graycoprops(graycom, 'ASM')
    # print all features
    ## return as a list
    return contrast.flatten() + dissimilarity.flatten() + homogeneity.flatten() + energy.flatten() + correlation.flatten() + ASM.flatten()

def siftExtraction(img):
  sift = SIFT()
  ## scale image to 1/3 of original size
  img2 = cv2.resize(img, (650, 1000))
  sift.detect_and_extract(img2)
  #print("SIFT features: {}".format(sift.descriptors))
  return sift.descriptors.flatten()

def hogFeature(img):
  fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, feature_vector=True)
  #print("HOG features: {}".format(fd))
  return fd 

def pcaReduction(featureList):
  pca = PCA()
  data = pca.pca.fit_transform(featureList)
  #print("PCA features: {}".format(pca.singular_values_))
  return data

def generateFeatureFile():
    currentPath = os.getcwd() + "/database/dataset_cropped"
    files = os.listdir(currentPath)
    counter = 0
    featureList = []
    y = []
    for file in files:
      if counter == 2:
        break
      counter += 1
      #append label, last three characters of file name and remove the extension
      y.append(file[-10:-7])
      # read image
      rootImg = cv2.imread(currentPath + "/" + file)
      img = cv2.cvtColor(rootImg, cv2.COLOR_BGR2GRAY)
      #extract features
      lbp = lbpExtraction(img)
      glcm = glcmExtraction(img)
      sift = siftExtraction(img)
      img2 = resize(img, (128, 64))
      hog = hogFeature(img2)
      # create feature list
      newFeatureList = np.concatenate((lbp, glcm, sift, hog))
      ## print features
      print("LBP features: {}".format(len(lbp)))
      print("GLCM features: {}".format(len(glcm)))
      print("SIFT features: {}".format(len(sift)))
      print("HOG features: {}".format(len(hog)))
      featureList.append(newFeatureList)
      print("Feature extraction for " + file + " completed")
      print(len(newFeatureList))
    # get y value of feature list 
    #x = StandardScaler().fit(featureList)
    #print(x)
    #print(len(x))

    #pca = pcaReduction(featureList.transpose())

    # file = open("feature.txt", "w+")
    # # write each feature in a line
    # for feature in featureList:
    #     file.write(str(feature) + " ")
    # file.write(str(feature) + " 0001")
    # file.close()




def main():
    generateFeatureFile()



if __name__=="__main__":
    main()