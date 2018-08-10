import matplotlib.pyplot as plt


def plot(title_lst, loss_lst, accuracy_lst, filename, figsize=(15, 10), all_in_one=False):
    """
    Create separate subplots for every loss and accuracy in the provided lists against the number of training epochs.
    The subplots will have the corresponding titles from title_lst.

    :param title_lst: a list of strings containing the titles for the subplots. One title for each training session. If
    all_in_one plot option is used, then title_lst functions as a list of labels for the legend.
    :param loss_lst: a 2d list: a list containing loss_logs from various training sessions
    :param accuracy_lst: a 2d list: a list containing accuracy_logs from various training sessions
    :param filename: filename under which the plot will be saved. If all_in_one plot option is used, 'comparision' will
    be added to the filename
    :param figsize: the size of the generated plot
    :param all_in_one: If True, then all_in_one plot option is enabled. Default: False
    :return: None
    """

    num_train_sessions = len(loss_lst)

    if all_in_one:
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # plot loss
        for i in range(num_train_sessions):
            epoch_lst = list(range(1, len(loss_lst[i]) + 1))
            ax[0, 0].plot(epoch_lst, loss_lst[i], label=title_lst[i])

        ax[0, 0].set_xlabel('Epoch')
        ax[0, 0].set_ylabel('Loss')
        ax[0, 0].set_title('Comparision of Losses')
        ax[0, 0].grid(True)
        ax[0, 0].legend()

        # plot accuracy
        for i in range(num_train_sessions):
            epoch_lst = list(range(1, len(accuracy_lst[i]) + 1))
            ax[0, 1].plot(epoch_lst, accuracy_lst[i], label=title_lst[i])

        ax[0, 1].set_xlabel('Epoch')
        ax[0, 1].set_ylabel('Accuracy')
        ax[0, 1].set_title('Comparision of Accuracies')
        ax[0, 1].grid(True)
        ax[0, 1].legend()

        filename += 'comparision'
    else:
        fig, ax = plt.subplots(num_train_sessions, 2, figsize=figsize)

        for i in range(num_train_sessions):
            epoch_lst = list(range(1, len(loss_lst[i]) + 1))

            # plot loss
            ax[i, 0].plot(epoch_lst, loss_lst[i], c='b')
            ax[i, 0].set_xlabel('Epoch')
            ax[i, 0].set_ylabel('Loss')
            ax[i, 0].set_title('Loss of ' + title_lst[i])
            ax[i, 0].grid(True)

            # plot accuracy
            ax[i, 1].plot(epoch_lst, accuracy_lst[i], c='b')
            ax[i, 1].set_xlabel('Epoch')
            ax[i, 1].set_ylabel('Accuracy')
            ax[i, 1].set_title('Accuracy of ' + title_lst[i])
            ax[i, 1].grid(True)

    plt.tight_layout()

    fig.savefig(filename + ".png")

    plt.show()
