# Deep Learning Challenge

Created by: Juan Camilo Pineda Iral.

Contanct information: camiloiral@gmail.com

**LENET5 MODIFIED**

The model implemented is a modification of the lenet5 model described in ([http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf))

The activation function used was "relu".
Activation is applied before pooling.
Max pooling pooling was used (instead of average pooling).
Batch normalization was applied on every convolutional layer.


The model calculates the mean "cross entropy" loss among the 43 classes.
The labels were one hot encoded.
The optimization algorithm used was the "Adam" optimizer.

The parameters used to train the model are the following:

Learning rate = 0.001

Batch size = 32

Epochs = 10

*IMPLEMENTATION*

model2: Modified Lenet5 architecture implemented in Tensorflow

To train a model type in terminal the following command:

    python app.py train -m [chosen mode] -d [directory with training data]

Example:

    python app.py train -m model2 -d /home/mypc/German_Traffic_Signs_Detector/images/train

To change any parameter for the training process type the command:

    python app.py train -m [chosen mode] -d [directory with training data] -option value

Example:

    python app.py train -m model2 -d /home/mypc/German_Traffic_Signs_Detector/images/train -lr 0.0001

To see more options type:

    python app.py train --help
