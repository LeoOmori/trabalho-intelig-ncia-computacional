import cv2
import numpy as np



def mask (image):
    ## remove noise from image
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


    # Find the coordinates of all white pixels in the image
    coords = cv2.findNonZero(opening )

    # Calculate the bounding box from the coordinates
    x, y, w, h = cv2.boundingRect(coords)

    # crop the image with the bounding_box
    cropped = image[y:y+h, x:x+w]
    
    # resize image to 2000x3000
    newImg = cv2.resize(cropped, (2000, 2000), interpolation=cv2.INTER_AREA)

    return newImg


# Load the image
image = cv2.imread('teste.bmp',cv2.IMREAD_GRAYSCALE)

mask(image)


