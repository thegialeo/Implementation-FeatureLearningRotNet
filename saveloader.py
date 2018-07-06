import pickle


def save_variable(var_list, filename):
    '''
    Takes a list of variables and save them in a .pkl file.

    :param var_list: a list of variables to save
    :param filename: name of the file the variables should be save in
    :return: None
    '''
    with open(filename, 'w') as f:
        pickle.dump(var_list, f)

