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

    subdir = "./plot"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    fig.savefig(os.path.join(subdir, filename + ".png"))
    
    plt.show()


def plot_all():
    """
    Create plots for loss and accuracies history of all experiments.

    :return: None
    """

    # load losses and accuracies
    rot_loss_3, rot_acc_3, clf_loss_3, clf_acc_3, conv_loss_3, conv_acc_3 = fm.load_variable("3_block_net")
    rot_loss_4, rot_acc_4, clf_loss_4, clf_acc_4, conv_loss_4, conv_acc_4 = fm.load_variable("4_block_net")
    rot_loss_5, rot_acc_5, clf_loss_5, clf_acc_5, conv_loss_5, conv_acc_5 = fm.load_variable("5_block_net")
    super_loss, super_acc = fm.load_variable("supervised_NIN")
    semi_loss, semi_acc, semi_sup_loss, semi_sup_acc = fm.load_variable("semi-supervised")



    # plot rotation task
    rot_titles = ["Rotation Task of 3 Block RotNet", "Rotation Task of 4 Block RotNet",
                  "Rotation Task of 5 Block RotNet"]
    rot_loss = [rot_loss_3, rot_loss_4, rot_loss_5]
    rot_acc = [rot_acc_3, rot_acc_4, rot_acc_5]
    plot(rot_titles, rot_loss, rot_acc, "Rotation Task")
    plot(rot_titles, rot_loss, rot_acc, "Rotation Task", all_in_one=True)

    # plot non-linear classifier
    clf_loss = [clf_loss_3, clf_loss_4, clf_loss_5]
    clf_acc = [clf_acc_3, clf_acc_4, clf_acc_5]
    for i in range(3, 6):
        clf_titles = []
        for j in range(1, i+1):
            clf_titles.append("Non-Linear Classifier trained on ConvBlock {}".format(j))
        plot(clf_titles, clf_loss[i - 3], clf_acc[i - 3], "Non-Linear Classifier and {} Block RotNet".format(i))
        plot(clf_titles, clf_loss[i - 3], clf_acc[i - 3], "Non-Linear Classifier and {} Block RotNet".format(i),
             all_in_one=True)

    # plot convolutional classifier
    conv_loss = [conv_loss_3, conv_loss_4, conv_loss_5]
    conv_acc = [conv_acc_3, conv_acc_4, conv_acc_5]
    for i in range(3, 6):
        conv_titles = []
        for j in range(1, i+1):
            conv_titles.append("ConvClassifier trained on ConvBlock {}".format(j))
        plot(conv_titles, conv_loss[i - 3], conv_acc[i - 3], "Convolutional Classifier and {} Block RotNet".format(i))
        plot(conv_titles, conv_loss[i - 3], conv_acc[i - 3], "Convolutional Classifier and {} Block RotNet".format(i),
             all_in_one=True)

    # plot supervised NIN
    plot(["Supervised NIN"], super_loss, super_acc, "Supervised NIN")
    plot(["Supervised NIN"], super_loss, super_acc, "Supervised NIN", all_in_one=True)

    # plot semi-supervised learning
    plot(["Semi-supervised", "Supervised NIN"], [semi_loss, semi_sup_loss], [semi_acc, semi_sup_acc],
         "Semi-supervised Learning")
    plot(["Semi-supervised", "Supervised NIN"], [semi_loss, semi_sup_loss], [semi_acc, semi_sup_acc],
         "Semi-supervised Learning", all_in_one=True)
