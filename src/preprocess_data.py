from tqdm import tqdm
from PIL import Image
import numpy as np
import click
import os
import cv2


def load_image(image_file):
    """
    Load RGB image from a file
    Inputs:
        image_file: image file to load
    Return:
        Image as numpy array with BGR format
    """
    try:
        img = Image.open(image_file)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    except IOError:
        print("File {} is not a supported image file".format(image_file))


def normalize_image(img):
    """
    Normalizes the image with min-max method
    Inputs:
        img: image to normalize
    Return:
        Normalized image
    """
    return cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)


def resize(img, img_size):
    """
    Resize the image to the specified img_size
    Inputs:
        img_size: New size of the image
    Return:
        Resized image. Shape (img_size, img_size, 3)
    """
    return cv2.resize(img, (img_size, img_size), cv2.INTER_AREA)


def translate_image(img, args):
    """
    Translates the image in x and y a percentage between 0 and 1
    Inputs:
        img: Image to translate
        args: Dictionary with "translate" argument
    Return:
        Image translated a percentage value in x and y axis
    """
    rows, cols, _ = img.shape
    M = np.float32([[1,0,int(cols*args["translate"])],[0,1,int(rows*args["translate"])]])

    return cv2.warpAffine(img,M,(cols,rows))


def rotate_image(img, args):
    """
    Rotates the image a given angle between 0 and 360 degrees
    Inputs:
        img: Image to rotate
        args: Dictionary with "angle" argument
    Return:
        Image rotated the given angle
    """
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), args["angle"], 1)

    return cv2.warpAffine(img, M,(cols, rows))


def flip(img, args):
    """
    Flips the image orientation
    Inputs:
        img: Image to flip
        args: Dictionary with "orientation" argument
    Return:
        Image flipped to the given orientation.
        "orientation" = 0, flips the image horizontally
    """
    return cv2.flip(img, args["orientation"])


def ligth_condition(img, args):
    """
    Change ligthning condition in the image
    Inputs:
        img: Image to change ligthning
        args: Dictionary with "gamma" argument
    Return:
        Image with ligthning values changed
    """
    invGamma = 1.0 / args["gamma"]
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)


def augment_data(img, file, functions, args):
    """
    Applies all the created image transformation functions
    and augments the passed image
    Inputs:
        img: Image to transform
        file: file name of the image
        functions: list with the transformation function operations
        args: Dictionary with the argument values if the function transformations
    Return:
        None
    """
    # Extract the class from the file name
    label, name = file.split("_")
    count = 0
    # Apply the image transformations and save the image changes
    for function in functions:
        image = function(img, args)
        image = resize(image, img_size=args["size"])
        cv2.imwrite(label+"_"+str(count)+"_"+name, image)
        count += 1


def load_dataset(d, img_size, mode = "train-test"):
    """
    Loads all the images
    Inputs:
        d: Directory with the images
        img_size: New size of the image
        mode: Mode to load the data. Values: "train-test" or "predict"
    Return:
        If mode = "train-test" return images and labels
        If mode = "predict" returns images only
    """
    os.chdir(d)
    files = os.listdir(d)

    if not mode in ["train-test", "predict"]:
        raise ValueError("No valid loading mode. Select between 'train-test' or 'predict' mode.")

    print("#============ Loading Data ============#")
    images = []
    labels = []

    # Load, resize and normalize the images from folder
    for file in tqdm(files):
        if os.path.isfile(file):
            img = load_image(file)
            img = resize(img, img_size)
            img = normalize_image(img)
            images.append(img)
            if mode == "train-test":
                # Extracts the label from the file name
                labels.append(int(file.split("_")[0]))

    if mode == "train-test":
        X, Y = np.array(images), np.array(labels)
        return X, Y
    elif mode == "predict":
        X = np.array(images)
        return X
