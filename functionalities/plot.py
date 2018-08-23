import matplotlib.pyplot as plt


def plot(title_lst, loss_lst, accuracy_lst, filename, figsize=(15, 10), all_in_one=False, max_accuracy=None,
         best_epoch=None):
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
    :param max_accuracy: If provided with best_epoch, the point (best_epoch, max_accuracy) will be additionally plotted
    :param best_epoch: If provided with max_accuracy, the point (best_epoch, max_accuracy) will be additionally plotted
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
            if best_epoch is not None and max_accuracy is not None:
                ax[i, 1].scatter(best_epoch[i], max_accuracy[i], c='r')
            ax[i, 1].set_xlabel('Epoch')
            ax[i, 1].set_ylabel('Accuracy')
            ax[i, 1].set_title('Accuracy of ' + title_lst[i])
            ax[i, 1].grid(True)

    plt.tight_layout()

    fig.savefig(filename + ".png")

    plt.show()


def plot_all(num_conv_block):
    """
    Create plots for loss and accuracies history of the RotNet model with the given number of convolutional blocks.

    :param num_conv_block: number of convolutional blocks in the RotNet model. If num_conv_block is 0, then Supervised
    NIN will be plotted instead.
    :return: None
    """

    if num_conv_block == 0:
        loss, _, accuracy, max_accuracy, best_epoch = fm.load_variable("RotNet_classification_200")
        plot(["Supervised NIN Classification"], [loss], [accuracy], "Supervised NIN Classification",
             max_accuracy=[max_accuracy], best_epoch=[best_epoch])
    else:
        loss = []
        accuracy = []
        max_accuracy = []
        best_epoch = []
        title = []

        conv_loss = []
        conv_accuracy = []
        conv_max_accuracy = []
        conv_best_epoch = []
        conv_title = []

        rot_loss, _, rot_accuracy, rot_max_accuracy, rot_best_epoch = fm.load_variable(
            "RotNet_rotation_200_{}_block_net".format(num_conv_block))

        plot(["Rotation Task with {} ConvBlock RotNet".format(num_conv_block)], [rot_loss], [rot_accuracy],
             "Rotation Task with {} ConvBlock RotNet".format(num_conv_block))

        for i in range(1, num_conv_block + 1):
            clf_loss, _, clf_accuracy, clf_max_accuracy, clf_best_epoch = fm.load_variable(
                "Classifier_block_{}_epoch_100_{}_block_net".format(i, num_conv_block))

            loss.append(clf_loss)
            accuracy.append(clf_accuracy)
            max_accuracy.append(clf_max_accuracy)
            best_epoch.append(clf_best_epoch)
            title.append("Classification Task ")

            conv_clf_loss, _, conv_clf_accuracy, conv_clf_max_accuracy, conv_clf_best_epoch = fm.load_variable(
                "ConvClassifier_block_{}_epoch_100_{}_block_net".format(i, num_conv_block))

            conv_loss.append(conv_clf_loss)
            conv_accuracy.append(conv_clf_accuracy)
            conv_max_accuracy.append(conv_clf_max_accuracy)
            conv_best_epoch.append(conv_clf_best_epoch)

        plot([])