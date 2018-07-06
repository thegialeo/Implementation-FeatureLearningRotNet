import pickle
import os
import sys


def save_variable(var_list, filename):
    '''
    Takes a list of variables and save them in a .pkl file.

    :param var_list: a list of variables to save
    :param filename: name of the file the variables should be save in
    :return: None
    '''

    subdir = "./variables"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    with open(os.path.join(subdir, filename + ".pkl"), 'w') as f:
        pickle.dump(var_list, f)


def load_variable(filename):
    '''
    Load variables from a .pkl file.

    :param filename: name of the file to load the variables from
    :return: list of variables loaded from .pkl file
    '''

    path = os.path.join("./variables", filename + ".pkl")

    with open(path) as f:
        var_list = pickle.load(f)

    return var_list