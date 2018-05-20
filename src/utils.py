import numpy as np
import itertools
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class_name = {
0 : 'speed limit 20 (prohibitory)',
1 : 'speed limit 30 (prohibitory)',
2 : 'speed limit 50 (prohibitory)',
3 : 'speed limit 60 (prohibitory)',
4 : 'speed limit 70 (prohibitory)',
5 : 'speed limit 80 (prohibitory)',
6 : 'restriction ends 80 (other)',
7 : 'speed limit 100 (prohibitory)',
8 : 'speed limit 120 (prohibitory)',
9 : 'no overtaking (prohibitory)',
10 : 'no overtaking (trucks) (prohibitory)',
11 : 'priority at next intersection (danger)',
12 : 'priority road (other)',
13 : 'give way (other)',
14 : 'stop (other)',
15 : 'no traffic both ways (prohibitory)',
16 : 'no trucks (prohibitory)',
17 : 'no entry (other)',
18 : 'danger (danger)',
19 : 'bend left (danger)',
20 : 'bend right (danger)',
21 : 'bend (danger)',
22 : 'uneven road (danger)',
23 : 'slippery road (danger)',
24 : 'road narrows (danger)',
25 : 'construction (danger)',
26 : 'traffic signal (danger)',
27 : 'pedestrian crossing (danger)',
28 : 'school crossing (danger)',
29 : 'cycles crossing (danger)',
30 : 'snow (danger)',
31 : 'animals (danger)',
32 : 'restriction ends (other)',
33 : 'go right (mandatory)',
34 : 'go left (mandatory)',
35 : 'go straight (mandatory)',
36 : 'go right or straight (mandatory)',
37 : 'go left or straight (mandatory)',
38 : 'keep right (mandatory)',
39 : 'keep left (mandatory)',
40 : 'roundabout (mandatory)',
41 : 'restriction ends (overtaking) (other)',
42 : 'restriction ends (overtaking (trucks)) (other)'
}


def plot_predictions(X_pred_images, preds):
    """
    Plot the images to predict with the predicted labels
    Inputs:
        X_pred_images: Images loaded from the seleted prediction folder
        preds: Predictions made by the model
    Return:
        None
    """
    # Convert images from BGR format to RGB
    images = np.array([cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) for img in X_pred_images])

    fig = plt.figure()

    # Iterator for displaying the next image
    image = itertools.cycle(images)
    pred = itertools.cycle(preds)

    # Plot the first image
    plt.imshow(next(image))
    label = next(pred)
    plt.text(15, 35, "Prediction: "+str(label)+" Class name: "+class_name[label],
            horizontalalignment='center', verticalalignment='center')
    plt.title("Press CLICK to see NEXT predicted image")

    def onclick(event):
        """ Plots the image with the corresponding prediction when click event happens"""
        plt.clf()
        plt.imshow(next(image))
        label = next(pred)
        plt.text(15, 35, "Prediction: "+str(label)+" Class name: "+class_name[label],
                horizontalalignment='center', verticalalignment='center')
        plt.title(" Press CLICK to see NEXT predicted image")
        plt.show()

    # Wait for click event to happen and executes onclick function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
