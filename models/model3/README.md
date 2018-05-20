# Deep Learning Challenge

Created by: Juan Camilo Pineda Iral.

Contanct information: camiloiral@gmail.com

**LOGISTIC REGRESSION MODEL**

The logistic regression model implemented in Tensorflow.

The model calculates the mean "cross entropy" loss among the 43 classes.
The labels were one hot encoded.
The optimization algorithm used was the "Adam" optimizer.

The parameters used to train the model are the following:

Learning rate = 0.001

Batch size = 32

Epochs = 20

*IMPLEMENTATION*

model3: Logistic regression architecture implemented in Tensorflow

To train a model type in terminal the following command:

    python app.py train -m [chosen mode] -d [directory with training data]

Example:

    python app.py train -m model3 -d /home/mypc/German_Traffic_Signs_Detector/images/train

To change any parameter for the training process type the command:

    python app.py train -m [chosen mode] -d [directory with training data] -option value

Example:

    python app.py train -m model3 -d /home/mypc/German_Traffic_Signs_Detector/images/train -lr 0.0001

To see more options type:

    python app.py train --help
