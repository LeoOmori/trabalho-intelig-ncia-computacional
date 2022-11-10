from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import cv2  
import numpy as np

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
    print("Histogram of Local Binary Pattern value: {}".format(hist))   
    return lbp

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
    print("Contrast: {} {} {} {} {} {}".format(contrast, dissimilarity, homogeneity, energy, correlation, ASM))

def main():
    img = cv2.imread("dataset_cartas/CF00314_02.bmp")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glcmExtraction(img)

if __name__=="__main__":
    main()