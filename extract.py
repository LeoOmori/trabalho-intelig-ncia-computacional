from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
import pandas as pd
import os
import cv2  
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np



def extract_sift_descriptors(image):
    sift =  cv2.SIFT_create()
    kp, descriptors = sift.detectAndCompute(image, None)
    return descriptors[-1]


def lbp_feature_extractor(img, radius, numPoints):
    # Compute the LBP image
    lbp_image = local_binary_pattern(img, numPoints, radius, method='nri_uniform')
    # Compute the histogram of the LBP image
    n_bins = int(lbp_image.max() + 1)
    (hist, _) = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    # Return the LBP features
    return hist.flatten()

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


def extract_orb_features(image):
    # Create ORB object
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp, descriptors = orb.detectAndCompute(image, None)

    newDescriptors = descriptors.flatten()
    return newDescriptors.flatten()


def generateFeatureFile():

    currentPath = os.getcwd() + "/database/dataset_processed"
    files = os.listdir(currentPath)
    counter = 0
    lbpFeatures = []
    glcmFeatures = []
    siftFeatures = []
    orbFeatures = []
    for file in files:
        #append in numpy like y.append(file[-10:-7])
        y = file[-10:-7]
        id = int(y)
        print(file[-10:-7])
        rootImg = cv2.imread(currentPath + "/" + file)
        img = cv2.cvtColor(rootImg, cv2.COLOR_BGR2GRAY)
        orb = extract_orb_features(img)
        lbp = lbp_feature_extractor(img, 2, 8)
        glcm = glcmExtraction(img)
        sift = extract_sift_descriptors(img)

        lbp = np.append(lbp, [id])
        glcm = np.append(glcm, [id])
        sift = np.append(sift, [id])
        orb = np.append(orb, [id])

        
        ## append to array    
        lbpFeatures.append(lbp)  
        glcmFeatures.append(glcm)
        siftFeatures.append(sift)
        orbFeatures.append(orb)


    lbpFeatures = np.array(lbpFeatures)
    glcmFeatures = np.array(glcmFeatures)
    siftFeatures = np.array(siftFeatures)
    orbFeatures = np.array(orbFeatures)


    # create a DataFrame to store the LBP features
    df_lbp = pd.DataFrame(lbpFeatures)
    # save the DataFrame to a CSV file
    df_lbp.to_csv("lbp_features2.csv", index=False)

    # create a DataFrame to store the GLCM features
    df_glcm = pd.DataFrame(glcmFeatures)
    # save the DataFrame to a CSV file
    df_glcm.to_csv("glcm_features2.csv", index=False)

    # create a DataFrame to store the SIFT features
    df_sift = pd.DataFrame(siftFeatures)
    # save the DataFrame to a CSV file
    df_sift.to_csv("sift_features2.csv", index=False)

    # create a DataFrame to store the ORB features
    df_orb = pd.DataFrame(orbFeatures)
    # save the DataFrame to a CSV file

    ## remove last column label, and save to another variable y
    y = df_orb.iloc[:, -1]
    df_orb = df_orb.iloc[:, :-1]

    ## standar the df_orb
    scaler = StandardScaler()
    scaler.fit(df_orb)
    df_orb = scaler.transform(df_orb)
    df_orb = pd.DataFrame(df_orb)
    
    pca = PCA(n_components=200)
    pca.fit(df_orb)
    df_orb = pca.transform(df_orb)
    df_orb = pd.DataFrame(df_orb)

    ## concat with pandas with y
    df_orb = pd.concat([df_orb, y], axis=1)
    # save the DataFrame to a CSV file
    df_orb.to_csv("orb_features2.csv", index=False)



def main():
    generateFeatureFile()

if __name__=="__main__":
    main()