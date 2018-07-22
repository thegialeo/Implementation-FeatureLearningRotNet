import os
import pickle
import torch


def save_variable(var_list, filename):
    """
    Takes a list of variables and save them in a .pkl file.

    :param var_list: a list of variables to save
    :param filename: name of the file the variables should be save in
    :return: None
    """

    subdir = "./variables"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    with open(os.path.join(subdir, filename + ".pkl"), 'w') as f:
        pickle.dump(var_list, f)


def load_variable(filename):
    """
    Load variables from a .pkl file.

    :param filename: name of the file to load the variables from
    :return: list of variables loaded from .pkl file
    """

    path = os.path.join("./variables", filename + ".pkl")

    with open(path) as f:
        var_list = pickle.load(f)

    return var_list


def save_net(net, filename):
    """
    Saves a neural network in a file.

    :param net: The neural network that should be saved.
    :param filename: name of the file the neural network should be save in
    :return: None
    """

    subdir = "./models"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    torch.save(net, os.path.join(subdir, filename))


def load_net(filename):
    """
    Loads a neural network from a file.

    :param filename: name of the file to load the neural network from
    :return: neural network from the file
    """

    path = os.path.join("./models", filename)

    net = torch.load(path)

    return net


def delete_file(path):
    """
    Delete the file corresponding to the given path.

    :param path: path of the file that should be deleted
    :return: None
    """

    if os.path.isfile(path):
        os.remove(path)


def add_block_to_name(num_block):
    """
    In the subdirectory ./models all saved nets and classifiers files created during training, will have
    (num_block)_block_net add to the end of their name.

    :param num_block: string. Intended to be the number of convolutional blocks in the RotNet
    :return: None
    """
    
