import numpy as np
import pandas as pd
import requests
import zipfile
import shutil
import click
import os
import io


def download_traffic_sign_data(zip_file_url, path, save_zip = False):
    """
    Downloads a zip file containig the German Traffic Sign Detection Benchmark
    and unzips the data cointained inside the folders of the zip file.
    Inputs:
        zip_file_url: url to the ziped data
        path: path to save the data
    Returns:
        String with current working directory
    """
    if os.path.exists(path+'/FullIJCNN2013'):
        print("Data already exists")
        return path
    elif os.path.exists(path+'/images/FullIJCNN2013'):
        print("Data already exists")
        return path+"/images"
    # Creates an "images" folder if it does not exist
    if not os.path.exists(path+"/images"):
        os.makedirs("images")

    os.chdir(path+"/images")
    print("Downloading images")

    # Downlods the zip file
    r = requests.get(zip_file_url)

    if save_zip:
        open('FullIJCNN2013.zip', 'wb').write(r.content) # Saves the downloaded zip
        print("Zip file successfully saved")

    zip = zipfile.ZipFile(io.BytesIO(r.content))
    all_files = zip.namelist()
    root_folder = all_files.pop(0) # Extracts the root folder 'FullIJCNN2013/'

    # Selects the files cointained inside the folders
    dirs = [file for file in all_files if not (root_folder+"00" in file) and \
            not (".txt" in file) or root_folder+"00/" in file]
    # Unzips the data
    for dir in dirs: zip.extract(dir)

    print("Data has been downloaded and unziped")

    return path+"/images"


def split_data(path, test_size= 0.2):
    """
    Splits the data into training and testing sets
    Inputs:
        path: Location of the Dataset folder
        test_size: Size of the test data (between 0 and 1)
    Return:
        train: path to the files corresponding to the train set
        test: path to the files corresponding to the test set
    """
    os.chdir(path)
    # Creates train, test and user directories if they don't exist
    if not os.path.exists(path+"/train"):
        os.makedirs("train")

    if not os.path.exists(path+"/test"):
        os.makedirs("test")

    if not os.path.exists(path+"/user"):
        os.makedirs("user")

    dataset  = os.path.join(path, "FullIJCNN2013")
    dirs = os.listdir(dataset)
    print("Splitting data")

    for dir in dirs:
        files = os.listdir(os.path.join(dataset, dir))
        num_files = len(files)
        # Select test data at random
        test_index = np.random.choice(num_files,int(num_files*test_size))

        select_test = [files[index] for index in test_index] # Extract test files
        select_train = [file for file in files if not (file in select_test)] # Extract train files
        # Move data to respective foldes
        test = [shutil.copy(os.path.join(dataset, dir, file), os.path.join(path, "test")+"/"+dir+"_"+file) \
            for file in select_test]
        train = [shutil.copy(os.path.join(dataset, dir, file), os.path.join(path, "train")+"/"+dir+"_"+file) \
                for file in select_train]

    shutil.rmtree(dataset) # Remove data directory
    print("Train and test data splitted correctly")

    return train, test


@click.group()
def main():
    pass


@main.command()
@click.option('-url', default="http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip",
              help='Url to download zip file', required=False)
@click.option('-d', default=os.getcwd(),
              help='Directory to download data')
@click.option('-save', default=False,
              help='Keep downloaded zip file')
@click.option('-test_size', default=0.2,
              help='Percentage of data to keep')
def download(url, d, save, test_size):
    """ Downloads and unzips the traffic sign detection  dataset"""
    path = download_traffic_sign_data(url, d, save)
    split_data(path, test_size)


if __name__ == '__main__':
    main()
