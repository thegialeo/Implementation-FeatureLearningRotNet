import os
import torch
import torch.optim as optim
import saveloader as sl
import rotation as rtt
import evaluater as eva


def train(num_epoch, net, trainloader, validloader, criterion, optimizer, classifier=None, conv_block_num=None,
          epoch_offset=0, rot=None, printing=True, max_accuracy=0, best_epoch=0, use_paper_metric=False):
    """
    Train a neural network.

    Optional: If rot is provided, the neural network is trained for rotation prediction task instead of the
    classification task (neural network is used for training).

    Optional: If a classifier and conv_block_num is provided, the classifier is attached to the feature map of
    the x-th convolutional block of the neural network (where x = conv_block_num) and trained for the classification
    task. Only the classifier is trained, not the neural network itself. No fine tuning.

    :param num_epoch: number of training epochs
    :param net: neural network that should be trained
    :param trainloader: the training set wrapped by a loader
    :param validloader: the validation set wrapped by a loader
    :param criterion: the criterion to compute the loss
    :param optimizer: the optimization method used for training
    :param classifier: optional argument, if provided, the classifier will be attached to the feature map of the x-th
    convolutional block of the neural network (where x = conv_block_num) and trained for classification task
    :param conv_block_num: number of the RotNet convolutional block to which the classifier will be attached
    :param epoch_offset: an offset to the training epoch numbers (useful, if this function is called several times for
    the same network)
    :param rot: list of classes for the rotation task. Possible classes are: '90', '180', '270'. Optional argument, if
    provided the neural network will be trained for the rotation task instead of the classification task.
    :param printing: if True, the max_accuracy and best_epoch will be additionally printed to the console
    :param max_accuracy: the highest accuracy achieved on the validation set so far
    :param best_epoch: the epoch in which the highest accuracy was achieved on the validation set
    :param use_paper_metric: use the metric from the paper "Unsupervised Representation Learning by Predicting Image
    Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis. Default: False
    :return: loss_log: a list of all losses computed at each training epoch
             accuracy_log: a list of all validation accuracies computed at each training epoch
             max_accuracy: the highest accuracy achieved on the validation set so far
             best_epoch: the epoch in which the highest accuracy was achieved on the validation set
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    if conv_block_num is not None:
        conv_block_num -= 1

    loss_log = []
    accuracy_log = []

    for epoch in range(epoch_offset, num_epoch + epoch_offset):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if rot is None:
                inputs, labels = inputs.to(device), labels.to(device)
                if classifier is None:
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    feats = net(inputs, out_feat_keys=[net.all_feat_names[conv_block_num]])
                    outputs = classifier(feats)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            else:
                rot_inputs, class_labels, rot_labels = rtt.create_rot_batch(inputs, labels, rot=rot)
                rot_inputs, rot_labels = rot_inputs.to(device), rot_labels.to(device)
                optimizer.zero_grad()
                outputs = net(rot_inputs)
                loss = criterion(outputs, rot_labels.long())
                loss.backward()
                optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 60 == 59:
                print('[{}, {}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 60))
                running_loss = 0.0

        # epoch loss
        loss_log.append(loss)
        print("Epoch: {} -> Loss: {}".format(epoch + 1, loss))

        # epoch validation accuracy
        accuracy = eva.get_accuracy(validloader, net, rot=rot, printing=False, classifier=classifier,
                                    conv_block_num=conv_block_num, use_paper_metric=use_paper_metric)
        accuracy_log.append(accuracy)
        print("Epoch: {} -> Evaluation Accuracy: {}".format(epoch + 1, accuracy))

        # save best model
        if accuracy >= max_accuracy:
            last_best_epoch = best_epoch
            best_epoch = epoch + 1
            max_accuracy = accuracy
            if rot is None:
                if classifier is None:
                    if use_paper_metric:
                        sl.save_net(net, 'RotNet_classification_{}_best_paper'.format(best_epoch))
                        if last_best_epoch != 0:
                            sl.delete_file('models/RotNet_classification_{}_best_paper'.format(last_best_epoch))
                    else:
                        sl.save_net(net, 'RotNet_classification_{}_best'.format(best_epoch))
                        if last_best_epoch != 0:
                            sl.delete_file('models/RotNet_classification_{}_best'.format(last_best_epoch))
                else:
                    if use_paper_metric:
                        sl.save_net(classifier, 'classifier_{}_best_paper'.format(best_epoch))
                        if last_best_epoch != 0:
                            sl.delete_file('models/classifier_{}_best_paper'.format(last_best_epoch))
                    else:
                        sl.save_net(classifier, 'classifier_{}_best'.format(best_epoch))
                        if last_best_epoch != 0:
                            sl.delete_file('models/classifier_{}_best'.format(last_best_epoch))
            else:
                if use_paper_metric:
                    sl.save_net(net, 'RotNet_rotation_{}_best_paper'.format(best_epoch))
                    if last_best_epoch != 0:
                        sl.delete_file('models/RotNet_rotation_{}_best_paper'.format(last_best_epoch))
                else:
                    sl.save_net(net, 'RotNet_rotation_{}_best'.format(best_epoch))
                    if last_best_epoch != 0:
                        sl.delete_file('models/RotNet_rotation_{}_best'.format(last_best_epoch))

    # printing
    if printing:
        print('highest validation accuracy: {:.3f} was achieved at epoch: {}'.format(max_accuracy, best_epoch))
        print('Finished Training')

    return loss_log, accuracy_log, max_accuracy, best_epoch


def adaptive_learning(lr_list, epoch_change, momentum, weight_decay, net, trainloader, validloader,
                      criterion, classifier=None, conv_block_num=None, rot=None, use_paper_metric=False):
    """
    Use adaptive learning rate to train the neural network.

    Optional: If rot is provided, the neural network is trained for rotation prediction task instead of the
    classification task (neural network is used for training).

    Optional: If a classifier and conv_block_num is provided, the classifier is attached to the feature map of
    the x-th convolutional block of the neural network (where x = conv_block_num) and trained for the classification
    task. Only the classifier is trained, not the neural network itself. No fine tuning.

    :param lr_list: a list of learning rates use for adaptive learning
    :param epoch_change: epochs where the learning rate should be change. Should have the same length as lr_list.
    :param momentum: momentum factor for stochastic gradient descent
    :param weight_decay: weight decay (L2 penalty) for stochastic gradient descent
    :param net: neural network that should be trained
    :param trainloader: the training set wrapped by a loader
    :param validloader: the validation set wrapped by a loader
    :param criterion: the criterion to compute the loss
    :param classifier: optional argument, if provided, the classifier will be attached to the feature map of the x-th
    convolutional block of the neural network (where x = conv_block_num) and trained for classification task
    :param conv_block_num: number of the RotNet convolutional block to which the classifier will be attached
    :param rot: list of classes for the rotation task. Possible classes are: '90', '180', '270'. Optional argument, if
    provided the neural network will be trained for the rotation task instead of the classification task.
    :param use_paper_metric: use the metric from the paper "Unsupervised Representation Learning by Predicting Image
    Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis. Default: False
    :return: loss_log: a list of all losses computed at each training epoch
             accuracy_log: a list of all validation accuracies computed at each training epoch
             max_accuracy: the highest accuracy achieved on the validation set so far
             best_epoch: the epoch in which the highest accuracy was achieved on the validation set
    """

    loss_log = []
    accuracy_log = []
    max_accuracy = 0
    best_epoch = 0

    for i, lr in enumerate(lr_list):
        if i == 0:
            epoch_offset = 0
        else:
            epoch_offset = epoch_change[i-1]

        num_epoch = epoch_change[i] - epoch_offset

        if classifier is None:
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        tmp_loss_log, tmp_accuracy_log, max_accuracy, best_epoch = train(num_epoch, net, trainloader, validloader,
                                                                         criterion, optimizer, classifier,
                                                                         conv_block_num, epoch_offset, rot, False,
                                                                         max_accuracy, best_epoch, use_paper_metric)

        loss_log += tmp_loss_log
        accuracy_log += tmp_accuracy_log

    return loss_log, accuracy_log, max_accuracy, best_epoch
