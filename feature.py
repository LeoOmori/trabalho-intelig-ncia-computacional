from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import cv2  
import numpy as np
from skimage.feature import SIFT
from skimage import transform
from skimage.feature import hog
from sklearn.decomposition import PCA

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

def siftExtraction(img):
  sift = SIFT()
  ## scale image to 1/3 of original size
  img2 = transform.rescale(img, 1.0 / 3.0)
  sift.detect_and_extract(img2)
  print("SIFT features: {}".format(sift.descriptors))

def hogFeature(img):
  fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, feature_vector=True)
  print("HOG features: {}".format(fd))

def pcaReduction(img):
  pca = PCA()
  pca.fit(img)
  print("PCA features: {}".format(pca.singular_values_))
  return 

def main():
    img = cv2.imread("\database\dataset_cartas\CF00001")
    print("Image shape: {}".format(img.singular_values_))
    hogFeature(img)

if __name__=="__main__":
    main()