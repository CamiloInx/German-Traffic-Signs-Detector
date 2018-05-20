import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pickle
import os
from preprocess_data import load_dataset

def train_sk_logistic(d):
    """
    Trains logistic regressoin model
    Inputs:
        d: Directory with the training data
    Return:
        None: saves the trained model and prints the training accuracy
    """
    path = os.getcwd()
    # Loading the dataset
    X_train_orig, Y_train = load_dataset(d, img_size=32, mode="train-test")
    n_values = len(set(Y_train))
    os.chdir(path)

    (m, n_H0, n_W0, n_C0) = X_train_orig.shape

    X_train = np.reshape(X_train_orig, (m, -1))

    print("Number of training examples = " + str(X_train.shape[0]))
    print("Total number of classes: ", n_values)
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))

    print("#============= Transforming data ============#")
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_transformed = pca.transform(X_train)

    clf = LogisticRegression()
    print("#============= Training model ============#")
    clf.fit(X_transformed, Y_train)

    if not os.path.exists(path+"/models/model4"):
        os.makedirs("models/model4/saved")

    save_dir = os.path.join(path, "models", "model4","saved")
    filename = os.path.join(save_dir, 'sk_logistic_regression.pckl')
    # save the model to disk
    log_models = [clf, pca]

    with open(filename, "wb") as f:
        for model in log_models:
            pickle.dump(model, f)

    print("#============= Model successfully saved =============#")

    train_preds = clf.predict(X_transformed)
    acc = accuracy_score(train_preds, Y_train)
    print("#============= Train accuracy: {} ============#".format(acc))

def evaluate_sk_logistic(d, mode="train"):
    """
    Evaluates the logistic regression trained model
    Inputs:
        d: Directory with the images
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
        X_test_orig, Y_test = load_dataset(d, img_size=32)
        (m, n_H0, n_W0, n_C0) = X_test_orig.shape

        X_test = np.reshape(X_test_orig, (m, -1))

        print ("Number of {}ing examples = {}".format(mode, str(X_test_orig.shape[0])))
        print ("X_{} shape: ".format(mode, str(X_test.shape)))
        print ("Y_{} shape: ".format(mode, str(Y_test.shape)))
    # Load images for prediction
    elif mode == "predict":
        # Load the dataset
        X_test_orig = load_dataset(d, img_size=32, mode="predict")
        (m, n_H0, n_W0, n_C0) = X_test_orig.shape

        X_test = np.reshape(X_test_orig, (m, -1))
        print ("Number of validation examples = " + str(X_test.shape[0]))
        print ("X_validation shape: " + str(X_test.shape))

    os.chdir(cwd)

    # Load the model
    checkp_file = os.path.join(cwd, "models", "model4", "saved", "sk_logistic_regression.pckl")
    models = []
    print("# ============Loading model. ==============#")

    with open(checkp_file, "rb") as f:
        [models.append(pickle.load(f)) for i in range(2)]

    clf, pca = models[:]
    # Transform data with pca
    X_transformed = pca.transform(X_test)
    preds = clf.predict(X_transformed) # Predict

    if mode == "train" or mode == "test":
        acc = accuracy_score(preds, Y_test)
        print("#============== {} accuracy: {} ==============#".format(mode, acc))
        return preds

    elif mode == "predict":
        return X_test_orig, preds
