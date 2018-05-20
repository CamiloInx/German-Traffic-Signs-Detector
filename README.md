# Deep Learning Challenge

Created by: Juan Camilo Pineda Iral.

Contanct information: camiloiral@gmail.com

This is a proposed solution for the first Deep Learning Challenge created by Kiwi Campus ([https://www.kiwicampus.com/](https://www.kiwicampus.com/)).

**DATABASE**

The database used for the challenge is [*The German Traffic Sign Detection Benchmark*](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) (GTSDB).

This database is created for image detection, but in this case it is used to perform image classification.
All the images in the database that where related to detection were omitted. The images used for this challenge are those that are inside the folders "00" to "42".

After selecting the images for the challenge we see that some of the classes have very few examples (i.e. only contains two or tree images), for that reason image augmentation techniques are used to help with this problem.

The images are divided in two folders: *"train"* and *"test"*. The images are splitted with 80% of them belonging to the *"train"* folder and 20% to the *"test"*. Image augmentation is only used on training images (to prevent overfitting).

The for creating and augmenting the images two main functions (*"download"* and *"augment"*) were developed.

For downloading the images type in terminal the following command:

    python app.py download

To see more options type:

    python app.py download --help

The download process may take a while, please be patient while downloading the database.
After downloading the database a *"images"* folder will be created and inside of it  *"train"*, *"test"* and *"user"* folders are also created. *"train"* folder will contain all the images for training the model. *"test"* folder will contain all the images for testing the model. *"user"* will contain the images to make predictions on. The names for the training and testing images are formatted in a specific way.

The  name format for the images inside the *"train"* and *"test"* folders is the following:

    08_04_00003.ppm
    00_00003.ppm

The first part of the name is the class of the image "08", then an " _ " separates the class of the image from the image name "04_00003.ppm". PLEASE KEEP THIS FORMAT WHEN ADDING IMAGES TO THE  *"train"* AND *"test"* FOLDERS. The *"user"* folder accepts any kind of format name.

For augmenting the images type in terminal the following command:

    python app.py augment

To change any parameter in the data augmentation process type the command:

    python app.py augment -option value

Example:

    python app.py augment -angle 65

To see more options type:

    python app.py augment --help

Keep in mind that data augmentation is only performed with the training data after splitting the database (80% training, 20% testing) to avoid overfitting.

**TRAINING**

The training operation is made with the data contained in the *"images/train/"* folder. The name format for the images is described in the section before.

The available models are:

model1: Original Lenet5 architecture implemented in Tensorflow ([http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf))

model2: Proposed Lenet5 architecture. "Sigmoid" activation changed for "relu" activation, uses max pooling instead of average pooling, pooling after activation, dropout and batch normalization added.

model3: Logistic regression implemented in tensorflow.

model4: Logistic regression implemented in sklearn

To train a model type in terminal the following command:

    python app.py train -m [chosen mode] -d [directory with training data]

Example:

    python app.py train -m model1 -d /home/mypc/German_Traffic_Signs_Detector/images/train

To change any parameter for the training process type the command:

    python app.py train -m [chosen mode] -d [directory with training data] -option value

Example:

    python app.py train -m model1 -d /home/mypc/German_Traffic_Signs_Detector/images/train -lr 0.0001

To see more options type:

    python app.py train --help


**TESTING**

The testing operation is made with the data contained in the *"images/test/"* folder. The name format for the images is described in the DATABASE section.

To test a model type in terminal the following command:

    python app.py test -m [chosen mode] -d [directory with testing data]

Example:

    python app.py test -m model1 -d /home/mypc/German_Traffic_Signs_Detector/images/test

The available models are:

model1: Original Lenet5 architecture implemented in Tensorflow ([http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf))

model2: Proposed Lenet5 architecture. "Sigmoid" activation changed for "relu" activation, uses max pooling instead of average pooling, pooling after activation, dropout and batch normalization added. (THIS MODEL WAS JUST TRAINED IN 10 EPOCHS, THE RESULTS ARE VERY POOR)

model3: Logistic regression implemented in tensorflow.

model4: Logistic regression implemented in sklearn.

To see more options type:

    python app.py test --help


**PREDICTION**

To predict an image or various images, place the images on the *"images/user/"* folder. The image name does NOT have to follow the naming convention described on the DATABASE section.

To predict an image or set of images type in terminal the following command:

    python app.py infer -m [chosen mode] -d [directory with image data]

Example:

    python app.py infer -m model1 -d /home/mypc/German_Traffic_Signs_Detector/images/user

The available models are:

model1: Original Lenet5 architecture implemented in Tensorflow ([http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf))

model2: Proposed Lenet5 architecture. "Sigmoid" activation changed for "relu" activation, uses max pooling instead of average pooling, pooling after activation, dropout and batch normalization added. (THIS MODEL WAS JUST TRAINED IN 10 EPOCHS, THE RESULTS ARE VERY POOR)

model3: Logistic regression implemented in tensorflow.

model4: Logistic regression implemented in sklearn.

To see more options type:

    python app.py infer --help

After predicting the images a window will display the original images with the predicted classes. If multiple images were passed, the window will allow to display all the images. To see next the image, just click on the window and the next image with the predicted class will be displayed.
