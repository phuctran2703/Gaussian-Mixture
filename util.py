import pickle
import numpy as np
import pandas as pd
import sys


def load_npy(file_name):
    """load_npy
    Load numpy data file. This is needed as python 2.7 pickle uses ascii as default encoding method but python 3.x uses utf-8.abs

    :param file_name: npy file path
    
    :return obj: loaded numpy object
    """
    
    if (sys.version_info[0] >= 3):
        obj = np.load(file_name, encoding='latin1')
    elif (sys.version_info[0] >=2):
        obj = np.load(file_name)
    
    return obj


def load_list(file_name):
    """load_list
    Load a list object to file_name.

    :param file_name: string, file name.
    """
    end_of_file = False
    list_obj = [] 
    f = open(file_name, 'rb')
    python_version = sys.version_info[0]
    while (not end_of_file):
        try:
            if (python_version >= 3):
                list_obj.append(pickle.load(f, encoding='latin1'))
            elif (python_version >=2):
                list_obj.append(pickle.load(f))
        except EOFError:
            end_of_file = True
            print("EOF Reached")

    f.close()
    return list_obj 


def save_list(list_obj, file_name):
    """save_list
    Save a list object to file_name
    
    :param list_obj: List of objects to be saved.
    :param file_name: file name.
    """

    f = open(file_name, 'wb')
    for obj in list_obj:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close() 

def get_iris_data(path="data/iris.dat"):
    """get_iris_data
    
    Load the data into 6 numpy arrays:
    * train_x1
    * train_x2
    * train_x3
    * test_x1
    * test_x2
    * test_x3
    :param path: path to the iris dataset file
    """ 
    print('Reading iris data...')
    dataIris = pd.read_table('data/iris.dat', sep='\\s+', header = None)
    dataIris.head()

    dataIris = dataIris.values
    dataIris[:, 4] = dataIris[:, 4]
    
    x1 = dataIris[dataIris[:,4]==1, 0:4]
    x2 = dataIris[dataIris[:,4]==2, 0:4]
    x3 = dataIris[dataIris[:,4]==3, 0:4]
    y1 = dataIris[dataIris[:,4]==1, 4]
    y2 = dataIris[dataIris[:,4]==2, 4]
    y3 = dataIris[dataIris[:,4]==3, 4]

    train_x1 = x1[0:40,:]
    test_x1 = x1[40:,:]
    train_x2 = x2[0:40,:]
    test_x2 = x2[40:,:]
    train_x3 = x3[0:40,:]
    test_x3 = x3[40:,:]

    train_y1 = y1[0:40].reshape((40,1))
    test_y1 = y1[40:].reshape((10,1))
    train_y2 = y2[0:40].reshape((40,1))
    test_y2 = y2[40:].reshape((10,1))
    train_y3 = y3[0:40].reshape((40,1))
    test_y3 = y3[40:].reshape((10,1))

    train_x = np.concatenate((train_x1, train_x2, train_x3), axis=0)
    train_y = np.concatenate((train_y1, train_y2, train_y3), axis=0)-1
    test_x = np.concatenate((test_x1, test_x2, test_x3), axis=0)
    test_y = np.concatenate((test_y1, test_y2, test_y3), axis=0)-1
    print("Done reading")
    return (train_x, train_y, test_x, test_y)


def normalize(train_x, val_x, test_x):
    """normalize
    This function computes train mean and standard deviation on all pixels then applying data scaling on train_x, val_x and test_x using these computed values

    :param train_x: train samples, shape=(num_train, num_feature)
    :param val_x: validation samples, shape=(num_val, num_feature)
    :param test_x: test samples, shape=(num_test, num_feature)
    """
    # train_mean and train_std should have the shape of (1, 1)
    train_mean = np.mean(train_x, axis=(0,1), dtype=np.float64, keepdims=True)
    train_std = np.std(train_x, axis=(0,1), dtype=np.float64, keepdims=True)

    train_x = (train_x-train_mean)/train_std
    val_x = (val_x-train_mean)/train_std
    test_x = (test_x-train_mean)/train_std
    return train_x, val_x, test_x


def create_one_hot(labels, num_k=10):
    """create_one_hot
    This function creates a one-hot (one-of-k) matrix based on the given labels

    :param labels: list of labels, each label is one of 0, 1, 2,... , num_k - 1
    :param num_k: number of classes we want to classify
    """
    labels = np.asarray(labels, dtype=int)
    eye_mat = np.eye(num_k)
    return eye_mat[labels, :].astype(np.float32)


def add_one(x):
    """add_one
    
    This function add ones as an additional feature for x
    :param x: input data
    """
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    return x


if __name__ == '__main__':
    get_iris_data()
