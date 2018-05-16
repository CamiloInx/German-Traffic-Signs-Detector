import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from preprocess_data import load_dataset

def lenet5(x, num_classes):
    # Input Layer
    # input_layer = tf.reshape(x, [-1, 32, 32, 3])
    input_layer = x

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
    dense1 = tf.layers.dense(inputs=conv3_flat, units=84)
    # Activation 4
    act4 = tf.sigmoid(dense1)
    # Output layer
    logits = tf.contrib.layers.fully_connected(act4, num_classes, activation_fn = tf.nn.softmax)

    return logits

def lenet5_improved(x, num_classes):
    # Input Layer
    input_layer = tf.reshape(x, [-1, 32, 32, 3])

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
    dense1 = tf.layers.dense(inputs=dropout, units=84)
    # Activation 4
    act4 = tf.nn.relu(dense1)
    # Output layer
    logits = tf.contrib.layers.fully_connected(act4, num_classes, activation_fn = tf.nn.softmax)

    return logits

def main(d, path=os.getcwd()):
    cwd = os.getcwd()
    # Loading the dataset
    X_train, Y_train_orig, classes = load_dataset(d, 32, 32)
    n_values = len(classes)
    os.chdir(cwd)
    # Normalize the vector
    # X_train = X_train_orig/255.0
    # X_test = X_test_orig/255.0
    # Convert the labels to one hot matrix
    Y_train = np.eye(n_values)[Y_train_orig[:]]
    # Y_test = np.eye(n_values)[Y_test_orig[0, :]]

    print ("number of training examples = " + str(X_train.shape[0]))
    # print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    # print ("X_test shape: " + str(X_test.shape))
    # print ("Y_test shape: " + str(Y_test.shape))

    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = n_values
    conv_layers = {}

    # Create placeholders
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0), name = "X")
    Y = tf.placeholder(tf.float32, shape = (None, n_y), name = "Y")
    batch_size = tf.placeholder(tf.int32)

    # Parameters for the optimizer
    EPOCHS = 50
    BATCH_SIZE = 32
    learning_rate = 1e-4
    #"decay": 1e-5,

    features, labels = (X, Y)

    dataset = tf.data.Dataset.from_tensor_slices((features,labels)).shuffle(buffer_size=100).repeat().batch(BATCH_SIZE)
    # dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)

    iter = dataset.make_initializable_iterator()
    x, y = iter.get_next()

    logits = lenet5(x, n_y)

    # Compute the cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y))

    # Create the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize variables
    init = tf.global_variables_initializer()

    costs = []
    acc = []
    print_cost = True
    num_minibatches = int(m / BATCH_SIZE)
    print("The total number of minibatches is: ", num_minibatches)

    saver = tf.train.Saver()

    if not os.path.exists(path+"/models"):
        os.makedirs("models")
    if not os.path.exists(path+"/models/model1"):
        os.makedirs("models/model1")

    save_dir = os.path.join(path, "models", "model1")
    save_path = os.path.join(save_dir, 'lenet5')

    # Train the neural netowrk
    with tf.Session() as sess:
        sess.run(init)
        sess.run(iter.initializer, feed_dict={ X: X_train, Y: Y_train, batch_size: BATCH_SIZE})

        for epoch in range(EPOCHS):
            epoch_cost = 0.
            epoch_acc = 0.
            print("Epoch:{}/{}".formtat(epoch, EPOCHS))
            for minibatch in tqdm(range(num_minibatches)):
                _, minibatch_cost = sess.run([optimizer, cost])
                epoch_cost += minibatch_cost / num_minibatches
                # Calculate the correct predictions
                # correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(y, 1))
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                epoch_acc = sess.run(accuracy)
                epoch_acc += epoch_acc / num_minibatches
            if print_cost == True and epoch % 1 == 0:
                print ("Cost:{}, Accuracy:{}".format(epoch_cost, epoch_acc))
                costs.append(epoch_cost)
                acc.append(epoch_acc)
            if print_cost == True and epoch % 10 == 0:
                saved_path = saver.save(sess=sess, save_path=save_path)
        saved_path = saver.save(sess=sess, save_path=save_path)
        print ("\nModel has been trained and saved in: ", saved_path)

        # print ("Train Accuracy:", sess.run(accuracy, feed_dict = {X: X_train, Y: Y_train}))
        # print ("Test Accuracy:", sess.run(accuracy, feed_dict = {X: X_test, Y: Y_test}))
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

if __name__ == '__main__':
    dir = "/home/camiloiral/Workspace/kiwi_challenge/German_Traffic_Signs_Detector/src/images/train"
    main(dir)
