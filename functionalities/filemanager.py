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


def add_block_to_name(num_block, num_epoch_lst, best_epoch_lst=None):
    """
    In the subdirectory ./models all saved nets and classifiers files created during training, will have
    (num_block)_block_net added to the end of their name.

    :param num_block: number. Intended to be the number of convolutional blocks in the RotNet
    :param num_epoch_lst: list of number of training epochs. This is needed to find the appropriate files
    :param best_epoch_lst: list of numbers. Optional: if provided, the best models saved during training will have
    (num_block)_block_net added to the end of their name as well
    :return: None
    """

    bib = ['RotNet_classification_{}_paper', 'RotNet_classification_{}', 'Classifier_block_{}_epoch_{}_paper',
           'ConvClassifier_block_{}_epoch_{}_paper', 'Classifier_block_{}_epoch_{}', 'ConvClassifier_block_{}_epoch_{}',
           'RotNet_rotation_{}_paper', 'RotNet_rotation_{}']

    names = []

    for i, string in enumerate(bib):
        if 2 <= i <= 5:
            for j in range(1, num_block + 1):
                for num_epoch in num_epoch_lst:
                    names.append(string.format(j, num_epoch))
                if best_epoch_lst is not None:
                    for best_epoch in best_epoch_lst:
                        names.append(string.format(j, best_epoch) + '_best')
        else:
            for num_epoch in num_epoch_lst:
                names.append(string.format(num_epoch))
            if best_epoch_lst is not None:
                for best_epoch in best_epoch_lst:
                    names.append(string.format(best_epoch) + '_best')

    for name in names:
        path_mod = os.path.join("./models", name)
        if os.path.isfile(path_mod):
            os.rename(path_mod, path_mod + '_{}_block_net'.format(num_block))
        base_path_var = os.path.join("./variables", name)
        path_var = base_path_var + ".pkl"
        if os.path.isfile(path_var):
            os.rename(path_var, base_path_var + '_{}_block_net.pkl'.format(num_block))

