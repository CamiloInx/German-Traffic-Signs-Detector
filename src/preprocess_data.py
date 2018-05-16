from tqdm import tqdm
import numpy as np
import click
import os
import cv2


# def show_image(img):
#     cv2.imshow('img', img)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()

def load_image(image_file):
    """
    Load RGB image from a file
    """
    return cv2.imread(image_file)

def normalize_image(img):
    return cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)

def resize(img, width, height):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(img, (width, height), cv2.INTER_AREA)

def translate_image(img, args):
    """Translates the image in x and y a percentage between 0 and 1"""
    rows, cols, _ = img.shape
    M = np.float32([[1,0,int(cols*args["translate"])],[0,1,int(rows*args["translate"])]])
    return cv2.warpAffine(img,M,(cols,rows))

def rotate_image(img, args):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), args["angle"], 1)
    return cv2.warpAffine(img, M,(cols, rows))

def flip(img, args):
    """ Flips the image orientation"""
    return cv2.flip(img, args["orientation"])

def ligth_condition(img, args):
    """ Change ligthning condition in the image"""
    invGamma = 1.0 / args["gamma"]
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def pass_image(img, args):
    return img

def augment_data(img, file, functions, args):
    label, name = file.split("_")
    count = 0
    for function in functions:
        image = function(img, args)
        image = resize(image, args["width"], args["height"])
        cv2.imwrite(label+"_"+str(count)+"_"+name, image)
        count += 1

def load_dataset(d, width, height):
    os.chdir(d)
    files = os.listdir(d)
    print("#============ Loading Data ============#")
    images = []
    labels = []
    for file in tqdm(files):
        if os.path.isfile(file) and ".ppm" in file:
            img = load_image(file)
            img = resize(img, width, height)
            img = normalize_image(img)
            images.append(img)
            labels.append(int(file.split("_")[0]))

    X, Y, classes = np.array(images), np.array(labels), set(labels)

    return X, Y, classes

@click.group()
def main():
    pass

@main.command()
@click.option('-d', default=os.getcwd(),
              help='Directory to augment images')
def augment(d):
    os.chdir(d)
    args = {
            "translate": 0.2,
            "angle": 45,
            "orientation": 0,
            "gamma": 1.5,
            "width": 32,
            "height": 32
            }
    files = os.listdir(d)
    functions = [translate_image, rotate_image, flip, ligth_condition, pass_image]
    print("#============ Augmenting Data ============#")
    for file in tqdm(files):
        if os.path.isfile(file) and ".ppm" in file:
            img = load_image(file)
            augment_data(img, file, functions, args)

if __name__ == '__main__':
    main()

# def add_noise(img):
#     row,col,ch= img.shape
#     mean = 0
#     var = 0.1
#     sigma = var**0.5
#     gauss = np.random.normal(mean,sigma,(row,col,ch))
#     gauss = gauss.reshape(row,col,ch)
#     noisy = (img + gauss)/255
#     show_image(noisy)

# Realizar data augmentation en los datos de entrenamiento
# Color shift or PCA color augmentation
# Add noise to image (pepper, gaussiang blur)
# Normalizar las imagenes
# Escribir la descripción de cada funcion
# Agregar todas las opciones para las entradas por la terminal
