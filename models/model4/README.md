# Deep Learning Challenge

Created by: Juan Camilo Pineda Iral.

Contanct information: camiloiral@gmail.com

**LOGISTIC REGRESSION MODEL**

The logistic regression model implemented with Sklearn.

PCA was used to reduce a little the amount of data.
PCA n_components = 0.95

The parametes for the classifier are:
regularizer = l2

C = 1.0

solver = liblinear

multi_class = ovr

*IMPLEMENTATION*

model4: Logistic regression architecture implemented in Sklearn

To train a model type in terminal the following command:

    python app.py train -m [chosen mode] -d [directory with training data]

Example:

    python app.py train -m model4 -d /home/mypc/German_Traffic_Signs_Detector/images/train

To see more options type:

    python app.py train --help
