import cv2
import os
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

def preprocess_image(image_path, output_path):
  # Load the image from the given path
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    print(image_path)
    # Apply global binarization using Otsu's algorithm
    threshold, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use the Sobel operator to detect contours
    contours = cv2.Sobel(binarized, cv2.CV_8U, 1, 0, ksize=3)
    # resize image to 2500x3500
    newImg = cv2.resize(contours, (2500, 3500), interpolation=cv2.INTER_AREA)
    cropped = mask(newImg)
    # Save the preprocessed image to the specified output path
    cv2.imwrite(output_path, cropped)


# Path to the directory containing the original images
input_dir = "database/dataset_cartas"

# Path to the directory where the preprocessed images should be saved
output_dir = "database/dataset_cropped"

# Create the output directory if it doesn't already exist
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

# Iterate over the files in the input directory
for filename in os.listdir(input_dir):  
    if filename.endswith(".bmp"):
        # Get the full path to the input and output files
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Preprocess the image and save the result
        preprocess_image(input_path, output_path)