import numpy as np
import itertools
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess_data import load_dataset

def lenet5(x, num_classes):
    """
    Creates the Lenet5 architecture described in http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    Inputs:
        x: Tensor with input images to the model
        num_classes: number of classes at the output
                     of the network (classes to predict)
    Return:
        logits: predictions for the input images
    """
    # Input Layer
    input_layer = tf.reshape(x, [-1, 32, 32, 3], name = "INPUT")

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=6, kernel_size=[5, 5], name ="C1",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Pooling Layer #1
    pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="S2")
    # Activation 1
    act1 = tf.sigmoid(pool1)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(inputs=act1, filters=16, kernel_size=[5, 5], name ="C3",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Pooling Layer #2
    pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="S4")
    # Activation 2
    act2 = tf.sigmoid(pool2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(inputs=act2, filters=120, kernel_size=[5, 5], name ="C5",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Activation 3
    act3 = tf.sigmoid(conv3)

    # Flatten Convolutional Layers
    conv3_flat = tf.contrib.layers.flatten(act3)
    # Dense Layer #1
    dense1 = tf.layers.dense(inputs=conv3_flat, units=84, name="F6",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Activation 4
    act4 = tf.sigmoid(dense1)
    # Output layer
    # logits = tf.contrib.layers.fully_connected(act4, num_classes, activation_fn = tf.nn.softmax)
    logits = tf.layers.dense(inputs=act4, units=num_classes, activation=tf.nn.softmax, name="OUTPUT",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))

    return logits


def lenet5_improved(x, num_classes):
    """
    Creates a modified version of the Lenet5 architecture
    described in http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    This function uses "relu" activation instead of "sigmoid",
    dropout and batch nomrmalization is added,
    activation is applied before pooling,
    uses max pooling instead of average pooling.
    Inputs:
        x: Tensor with input images to the model
        num_classes: number of classes at the output
                     of the network (classes to predict)
    Return:
        logits: predictions for the input images
    """
    # Input Layer
    input_layer = tf.reshape(x, [-1, 32, 32, 3], name = "INPUT")

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=6, kernel_size=[5, 5], name ="C1",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Nomrmalization #1
    norm1 = tf.layers.batch_normalization(conv1)
    # Activation #1
    act1 = tf.nn.relu(norm1)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=act1, pool_size=[2, 2], strides=2, name="S2")

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[5, 5], name ="C3",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Nomrmalization #2
    norm2 = tf.layers.batch_normalization(conv2)
    # Activation #2
    act2 = tf.nn.relu(norm2)
    # Pooling Layer #1
    pool2 = tf.layers.max_pooling2d(inputs=act2, pool_size=[2, 2], strides=2, name="S4")

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(inputs=pool2, filters=120, kernel_size=[5, 5], name ="C5",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Nomrmalization #3
    norm3 = tf.layers.batch_normalization(conv3)
    # Activation #3
    act3 = tf.nn.relu(norm3)

    # Flatten Convolutional Layers
    conv3_flat = tf.contrib.layers.flatten(act3)
    # Dropout
    dropout = tf.nn.dropout(conv3_flat, keep_prob=0.5)
    # Dense Layer #1
    dense1 = tf.layers.dense(inputs=dropout, units=84, name="F6",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
    # Nomrmalization #4
    norm4 = tf.layers.batch_normalization(dense1)
    # Activation 4
    act4 = tf.nn.relu(norm4)
    # Output layer
    logits = tf.layers.dense(inputs=act4, units=num_classes, activation=tf.nn.softmax, name="OUTPUT",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed = 0))

    return logits


def train_lenet5(d, model, learning_rate=1e-3, epochs=50, batch=32, print_cost = True):
    """
    Trains the Lenet5 model
    Inputs:
        d: Directory with the training data
        model: Select model to train. Models: model1-original_lenet5
                model2-improved_lenet5
        learning_rate: learning eate for the optimizer
        epochs: Total number of opochs for trainig
        batch: Mini-Batch size
        print_cost: Print the loss and accuracy in every epoch
    Return:
        Number of classes and saves the trained model
    """
    path = os.getcwd()
    # Loading the dataset
    X_train, Y_train_orig = load_dataset(d, img_size=32, mode="train-test")
    n_values = len(set(Y_train_orig))
    os.chdir(path)

    # Convert the labels to one hot matrix
    Y_train = np.eye(n_values)[Y_train_orig[:]]

    print("Number of training examples = " + str(X_train.shape[0]))
    print("Total number of classes: ", n_values)
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))

    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = n_values

    # Create placeholders
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0), name = "X")
    Y = tf.placeholder(tf.float32, shape = (None, n_y), name = "Y")
    batch_size = tf.placeholder(tf.int32)

    features, labels = (X, Y)

    dataset = tf.data.Dataset.from_tensor_slices((features,labels)).shuffle(buffer_size=1000).repeat().batch(batch)
    # dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)

    # Creates and interator for the traning data
    iter = dataset.make_initializable_iterator()
    x, y = iter.get_next()

    # Create model folder
    if not os.path.exists(path+"/models"):
        os.makedirs("models")

    # Model architecture
    if model == "model1":
        logits = lenet5(x, n_y)
        if not os.path.exists(path+"/models/model1"):
            os.makedirs("models/model1/saved")
        save_dir = os.path.join(path, "models", "model1","saved")
        save_path = os.path.join(save_dir, 'lenet5')
    elif model == "model2":
        logits = lenet5_improved(x, n_y)
        if not os.path.exists(path+"/models/model2"):
            os.makedirs("models/model2/saved")
        save_dir = os.path.join(path, "models", "model2","saved")
        save_path = os.path.join(save_dir, 'lenet5_improved')
    else:
        print("Model {} is not a valid model".format(m))

    # Compute the cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y))

    # Create the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize variables
    init = tf.global_variables_initializer()

    costs = []
    acc = []
    num_minibatches = int(m / batch)+1

    print("#============ Training model ============#")
    print("The total number of minibatches is: ", num_minibatches)

    saver = tf.train.Saver()

    # Train the neural netowrk
    with tf.Session() as sess:
        sess.run(init)
        sess.run(iter.initializer, feed_dict={ X: X_train, Y: Y_train, batch_size: batch})

        for epoch in range(epochs):
            epoch_cost = 0.
            epoch_acc = 0.
            print("Epoch:{}/{}".format(epoch, epochs))

            for minibatch in tqdm(range(num_minibatches)):
                _, minibatch_cost = sess.run([optimizer, cost])
                epoch_cost += minibatch_cost / num_minibatches

                # Calculate the correct predictions
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                epoch_acc = sess.run(accuracy)
                epoch_acc += epoch_acc / num_minibatches

            if print_cost == True and epoch % 1 == 0:
                print ("Loss:{}, Accuracy:{}".format(epoch_cost, epoch_acc))
                costs.append(epoch_cost)
                acc.append(epoch_acc)

            if print_cost == True and epoch % 10 == 0:
                saver.save(sess=sess, save_path=save_path) # Save the model

        saver.save(sess=sess, save_path=save_path)
        print ("\nModel has been trained and saved in: ", save_path)

    # if print_cost:
    #     # Plot the loss
    #     plt.plot(np.squeeze(costs))
    #     plt.ylabel('Loss')
    #     plt.xlabel('Iterations')
    #     plt.title("Learning rate =" + str(learning_rate))
    #     plt.show()

    return n_values


def evaluate_lenet5(d, model, n_classes, mode="train"):
    """
    Evaluates the Lenet5 trained model
    Inputs:
        d: Directory with the images
        model: Select model to train. Models: model1-original_lenet5
                model2-improved_lenet5
        n_classes= total number of classes to predict
                    (same as the number of outputs in the final layer of the model)
        mode: Mode to evaluate the data. Values: "train-test" or "predict"
    Return:
        If mode = "train", "test" return predictions
                    and prints the accuracy of the evaluation
        If mode = "predict" returns loaded images from prediction
                    folder and the predictions made by the Lenet5 model
    """
    cwd = os.getcwd()

    if not mode in ["train", "test", "predict"]:
        raise ValueError("No valid loading mode. Select between 'train-test' or 'predict' mode.")
    # If evaluating on the test data, load images and labels
    elif mode == "train" or mode == "test":
        # Load the dataset
        X_test, Y_test_orig = load_dataset(d, img_size=32)
        (m, n_H0, n_W0, n_C0) = X_test.shape
        Y_test = np.eye(n_classes)[Y_test_orig[:]] # Converto to one hot

        print ("Number of {}ing examples = {}".format(mode, str(X_test.shape[0])))
        print ("X_{} shape: {}".format(mode, str(X_test.shape)))
        print ("Y_{} shape: {}".format(mode, str(Y_test.shape)))

    # Load images for prediction
    elif mode == "predict":
        # Load the dataset
        X_pred = load_dataset(d, img_size=32, mode="predict")
        (m, n_H0, n_W0, n_C0) = X_pred.shape
        print ("Number of validation examples = " + str(X_pred.shape[0]))
        print ("X_validation shape: " + str(X_pred.shape))

    os.chdir(cwd)

    # Create placeholders
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0), name = "X")
    Y = tf.placeholder(tf.float32, shape = (None, n_classes), name = "Y")

    # ======= Start Lenet5 Model ======#
    if model == "model1":
        checkp_file = os.path.join(cwd, "models", "model1", "saved", "lenet5")
        logits = lenet5(X, n_classes)
    elif model == "model2":
        checkp_file = os.path.join(cwd, "models", "model2", "saved", "lenet5_improved")
        logits = lenet5_improved(X, n_classes)
    else:
        print("Model {} is not a valid model".format(model))
    # ======= END Lenet5 Model ========#

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkp_file) # Restore model

        if mode == "train" or mode == "test":
            # Calculate accuracy
            preds = tf.argmax(logits, 1)
            predictions = sess.run(preds, feed_dict={X: X_test})
            correct_prediction = tf.equal(preds, tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            acc = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
            print("#============ {} accuracy: {} ============#".format(mode, acc))
            return predictions

        elif mode == "predict":
            # Predict images
            preds = tf.argmax(logits, 1)
            predictions = sess.run(preds, feed_dict={X: X_pred})
            return X_pred, predictions
