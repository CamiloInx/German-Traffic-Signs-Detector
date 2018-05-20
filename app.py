import tensorflow as tf
import numpy as np
import itertools
import requests
import zipfile
import shutil
import click
import sys
import cv2
import os
import io
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from tensorflow.python.framework import ops

sys.path.append("./src/")
from download_data import *
from preprocess_data import *
from lenet5 import *
from utils import *
from tf_logistic import *
from sk_logistic import *

@click.group()
def main():
    """
    This is my solution for the first Kiwi Campus challenge \n
    Created by: Juan Camilo Pineda Iral \n
    Contact: camiloiral@gmail.com \n
    """
    pass


@main.command()
@click.option('-url', default="http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip",
              help='Url to download the zip file.', required=False)
@click.option('-d', default=os.getcwd(),
              help='Directory to download data.')
@click.option('-save', default=True, type=bool,
              help='Keep downloaded zip file.')
@click.option('-test_size', default=0.2,
              help='Percentage of data to use for testing. Values: 0.0-1.0')
def download(url, d, save, test_size):
    """ Downloads and unzips the German traffic signs detection dataset"""
    path = download_traffic_sign_data(url, d, save)
    split_data(path, test_size)


@main.command()
@click.option('-d', default=os.path.join(os.getcwd(), "images", "train"),
              help='Directory with images to augment.')
@click.option('-size', default=32,
              help='Change the size of the image: (size, size)')
@click.option('-translate', default=0.2,
              help='Translate the image in x and y. Values between: 0.0-1.0')
@click.option('-angle', default=45,
              help='Rotate the image the angle specified. Values between: 0-360')
@click.option('-orientation', default=0,
              help='Flip the image. Value 0-horizontal flip.')
@click.option('-gamma', default=1.5,
              help='Changes the brightness of the image. Values between: 0.0-2.0')
def augment(d, size, translate, angle, orientation, gamma):
    """
    Augment the number of images for training.
    """
    os.chdir(d)
    args = {
            "size": size,
            "translate": translate,
            "angle": angle,
            "orientation": orientation,
            "gamma": gamma
            }
    files = os.listdir(d)

    functions = [translate_image, rotate_image, flip, ligth_condition]
    print("#============ Augmenting Data ============#")

    for file in tqdm(files):
        if os.path.isfile(file):
            img = load_image(file)
            augment_data(img, file, functions, args)


@main.command()
@click.option('-m', default="model1", help='ML model to train. Models: model1, model2, ...')
@click.option('-d', default=os.path.join(os.getcwd(), "images", "train"),
              help='Directory with trainig data.')
@click.option('-lr', default=1e-3,
              help='Learning rate for the optimizer.')
@click.option('-epochs', default=50,
              help='Number of epochs to train the model on.')
@click.option('-batch', default=32,
              help='Mini-batch size for training the model.')
@click.option('-print_cost', default=True, type=bool,
              help='Print the cost and the accuracy for every minibatch.')
def train(m, d, lr, epochs, batch, print_cost):
    """Trains the selected model on the training data."""
    if m == "model1":
        classes = train_lenet5(d, model=m, learning_rate=lr, epochs=epochs, batch=batch, print_cost=print_cost)
        tf.reset_default_graph()
        evaluate_lenet5(d, model=m, n_classes= classes, mode="train")
    elif m == "model2":
        classes = train_lenet5(d, model=m, learning_rate=lr, epochs=epochs, batch=batch, print_cost=print_cost)
        tf.reset_default_graph()
        evaluate_lenet5(d, model=m, n_classes= classes, mode="train")
    elif m == "model3":
        classes = train_logistic(d, learning_rate=lr, epochs=epochs, batch=batch, print_cost=print_cost)
        tf.reset_default_graph()
        evaluate_logistic(d, n_classes= classes, mode="train")
    elif m == "model4":
        classes = train_sk_logistic(d)
        # evaluate_logistic(d, n_classes= classes, mode="train")
    else:
        print("Model '{}' is not a valid model".format(m))


@main.command()
@click.option('-m', default="model1", help='ML model to test. Models: model1, model2, ...')
@click.option('-d', default=os.path.join(os.getcwd(), "images", "test"),
              help='Directory with testing data.')
@click.option('-classes', default=43, help='Total number classes.')
def test(m, d, classes):
    """Tests the selected model on the testing data."""
    if m == "model1":
        evaluate_lenet5(d, model=m, n_classes= classes, mode="test")
    elif m == "model2":
        evaluate_lenet5(d, model=m, n_classes= classes, mode="test")
    elif m == "model3":
        evaluate_logistic(d, n_classes= classes, mode="test")
    elif m == "model4":
        evaluate_sk_logistic(d, mode="test")
    else:
        print("Model {} is not a valid model".format(m))


@main.command()
@click.option('-m', default="mod    el1", help='Predict image class with selected ML model. Models: model1, model2, ...')
@click.option('-d', default=os.path.join(os.getcwd(), "images", "user"),
              help='Directory with images to predict.')
@click.option('-classes', default=43, help='Total number classes.')
def infer(m, d, classes):
    """ Predicts the classes of the passed images."""
    if m == "model1":
        X_pred_images, preds = evaluate_lenet5(d, model=m, n_classes= classes, mode="predict")
        plot_predictions(X_pred_images, preds)
    elif m == "model2":
        X_pred_images, preds = evaluate_lenet5(d, model=m, n_classes= classes, mode="predict")
        plot_predictions(X_pred_images, preds)
    elif m == "model3":
        X_pred_images, preds = evaluate_logistic(d, n_classes= classes, mode="predict")
        plot_predictions(X_pred_images, preds)
    elif m == "model4":
        X_pred_images, preds = evaluate_sk_logistic(d, mode="predict")
        plot_predictions(X_pred_images, preds)
    else:
        print("Model {} is not a valid model".format(m))


if __name__ == '__main__':
    main()
