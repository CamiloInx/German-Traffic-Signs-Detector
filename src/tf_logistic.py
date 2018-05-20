import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess_data import load_dataset


def logistic_regression(x, num_classes):
    input = tf.reshape(x, [-1, 32*32*3], name = "INPUT")
    drop = tf.layers.dropout(input, rate = 0.2)
    w = tf.Variable(tf.random_normal(shape=[32*32*3,num_classes], stddev=0.01), name="weights")
    b = tf.Variable(tf.zeros([1,num_classes]), name="bias")
    logits = tf.matmul(drop, w) + b

    return logits


def train_logistic(d, learning_rate=1e-3, epochs=50, batch=32, print_cost = True):
    """
    Trains logistic regressoin model
    Inputs:
        d: Directory with the training data
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

    dataset = tf.data.Dataset.from_tensor_slices((features,labels)).shuffle(buffer_size=1000,).repeat().batch(batch)
    # dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)

    # Creates and interator for the traning data
    iter = dataset.make_initializable_iterator()
    x, y = iter.get_next()

    # Create model folder
    if not os.path.exists(path+"/models"):
        os.makedirs("models")

    # Model architecture
    logits = logistic_regression(x, n_y)

    if not os.path.exists(path+"/models/model3"):
        os.makedirs("models/model3/saved")

    save_dir = os.path.join(path, "models", "model3","saved")
    save_path = os.path.join(save_dir, 'logistic_regression')

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


def evaluate_logistic(d, n_classes, mode="train"):
    """
    Evaluates the logistic regression trained model
    Inputs:
        d: Directory with the images
        n_classes= total number of classes to predict
                    (same as the number of outputs in the final layer of the model)
        mode: Mode to evaluate the data. Values: "train", "test" or "predict"
    Return:
        If mode = "train"-"test", return predictions
                    and prints the accuracy of the evaluation
        If mode = "predict", returns loaded images from prediction
                    folder and the predictions made by the model
    """
    cwd = os.getcwd()

    if not mode in ["train","test", "predict"]:
        raise ValueError("No valid loading mode. Select between 'train', 'test' or 'predict' mode.")
    # If evaluating on the test data, load images and labels
    elif mode == "train" or mode == "test":
        # Load the dataset
        X_test, Y_test_orig = load_dataset(d, img_size=32)
        (m, n_H0, n_W0, n_C0) = X_test.shape
        Y_test = np.eye(n_classes)[Y_test_orig[:]] # Converto to one hot

        print ("Number of {}ing examples = {}".format(mode, str(X_test.shape[0])))
        print ("X_{} shape: ".format(mode, str(X_test.shape)))
        print ("Y_{} shape: ".format(mode, str(Y_test.shape)))
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

    # ======= Start Logistic Regression Model ======#
    checkp_file = os.path.join(cwd, "models", "model3", "saved", "logistic_regression")
    logits = logistic_regression(X, n_classes)

    # ======= END Logistic Regression Model ========#

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
