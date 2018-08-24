import os
import torch
import torch.optim as optim
from functionalities import dataloader as dl
from functionalities import filemanager as fm
from functionalities import rotation as rtt
from functionalities import evaluater as eva
from architecture import NonLinearClassifier as NLC
from architecture import ConvClassifier as CC
from architecture import RotNet as RN


def train(num_epoch, net, criterion, optimizer, trainloader, validloader=None, testloader=None, classifier=None,
          conv_block_num=None, epoch_offset=0, rot=None, printing=True, max_accuracy=0, best_epoch=0,
          use_paper_metric=False, use_ConvClassifier=False, semi=None):
    """
    Train a neural network.

    Optional: If rot is provided, the neural network is trained for rotation prediction task instead of the
    classification task (neural network is used for training).

    Optional: If a classifier and conv_block_num is provided, the classifier is attached to the feature map of
    the x-th convolutional block of the neural network (where x = conv_block_num) and trained for the classification
    task. Only the classifier is trained, not the neural network itself. No fine tuning.

    Optional: If a validation loader is provided, the validation accuracy will be evaluated after every training epoch
    and the best model will be kept.

    Optional: If a test loader is provided, the test accuracy will be evaluated after every training epoch

    :param num_epoch: number of training epochs
    :param net: neural network that should be trained
    :param criterion: the criterion to compute the loss
    :param optimizer: the optimization method used for training
    :param trainloader: the training set wrapped by a loader
    :param validloader: the validation set wrapped by a loader
    :param testloader: the test set wrapped by a loader
    :param classifier: optional argument, if provided, the classifier will be attached to the feature map of the x-th
    convolutional block of the neural network (where x = conv_block_num) and trained for classification task
    :param conv_block_num: number of the RotNet convolutional block to which the classifier will be attached
    :param epoch_offset: an offset to the training epoch numbers (useful, if this function is called several times for
    the same network)
    :param rot: list of classes for the rotation task. Possible classes are: '90', '180', '270'. Optional argument, if
    provided the neural network will be trained for the rotation task instead of the classification task.
    :param printing: if True, the max_accuracy and best_epoch will be additionally printed to the console. Also the
    current models will be saved.
    :param max_accuracy: the highest accuracy achieved on the validation set so far
    :param best_epoch: the epoch in which the highest accuracy was achieved on the validation set
    :param use_paper_metric: use the metric from the paper "Unsupervised Representation Learning by Predicting Image
    Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis. Default: False
    :param use_ConvClassifier: This parameter is not indented to be changed, but rather will be passed from the
    train_all_blocks function and subsequently from the adaptive_learning function. In other words, you do not need to
    provide this argument if using this function.
    :param semi: This parameter is not indented to be changed, but rather will be passed from the train_semi function.
    In other words, you do not need to provide this argument if using this function.
    :return: loss_log: a list of all losses computed at each training epoch
             accuracy_log: a list of all validation/test accuracies computed at each training epoch
             max_accuracy: the highest accuracy achieved on the validation set so far
             best_epoch: the epoch in which the highest accuracy was achieved on the validation set
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    if classifier is not None:
        classifier.to(device)

    if conv_block_num is not None:
        conv_block_num -= 1

    if use_paper_metric:
        paper_string = '_paper'
    else:
        paper_string = ''

    if use_ConvClassifier:
        conv_string = 'Conv'
    else:
        conv_string = ''

    if semi is not None:
        semi_string = 'Semi-supervised_{}_'.format(semi)
    else:
        semi_string = ''

    loss_log = []
    valid_accuracy_log = []
    test_accuracy_log = []

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

        if testloader is not None:
            # epoch test accuracy
            accuracy = eva.get_accuracy(testloader, net, rot=rot, printing=False, classifier=classifier,
                                        conv_block_num=conv_block_num if conv_block_num is None else conv_block_num + 1,
                                        use_paper_metric=use_paper_metric)
            test_accuracy_log.append(accuracy)
            print("Epoch: {} -> Test Accuracy: {}".format(epoch + 1, accuracy))

        if validloader is not None:

            # epoch validation accuracy
            accuracy = eva.get_accuracy(validloader, net, rot=rot, printing=False, classifier=classifier,
                                        conv_block_num=conv_block_num if conv_block_num is None else conv_block_num+1,
                                        use_paper_metric=use_paper_metric)
            valid_accuracy_log.append(accuracy)
            print("Epoch: {} -> Validation Accuracy: {}".format(epoch + 1, accuracy))

            # save best model
            if accuracy >= max_accuracy:
                last_best_epoch = best_epoch
                best_epoch = epoch + 1
                max_accuracy = accuracy
                if rot is None:
                    if classifier is None:
                        fm.save_net(net, '{}RotNet_classification_{}{}_best'.format(semi_string, best_epoch,
                                                                                    paper_string))
                        if last_best_epoch != 0:
                            fm.delete_file('models/{}RotNet_classification_{}{}_best'.format(semi_string,
                                last_best_epoch, paper_string))
                    else:
                        fm.save_net(classifier, '{}{}Classifier_block_{}_epoch_{}{}_best'.format(semi_string,
                            conv_string, conv_block_num + 1, best_epoch, paper_string))
                        if last_best_epoch != 0:
                            fm.delete_file('models/{}{}Classifier_block_{}_epoch_{}{}_best'.format(semi_string,
                                conv_string, conv_block_num + 1, last_best_epoch, paper_string))
                else:
                    fm.save_net(net, 'RotNet_rotation_{}{}_best'.format(best_epoch, paper_string))
                    if last_best_epoch != 0:
                        fm.delete_file('models/RotNet_rotation_{}{}_best'.format(last_best_epoch, paper_string))

    # printing
    if printing:
        if validloader is not None:
            print('highest validation accuracy: {:.3f} was achieved at epoch: {}'.format(max_accuracy, best_epoch))
        print('Finished Training')
        if rot is None:
            if classifier is None:
                fm.save_net(net, '{}RotNet_classification_{}{}'.format(semi_string, num_epoch + epoch_offset,
                                                                       paper_string))
                fm.save_variable([loss_log, valid_accuracy_log, test_accuracy_log, max_accuracy, best_epoch],
                    '{}RotNet_classification_{}{}'.format(semi_string, num_epoch + epoch_offset, paper_string))
            else:
                fm.save_net(classifier, '{}{}Classifier_block_{}_epoch_{}{}'.format(semi_string, conv_string,
                    conv_block_num + 1, num_epoch + epoch_offset, paper_string))
                fm.save_variable([loss_log, valid_accuracy_log, test_accuracy_log, max_accuracy, best_epoch],
                    '{}{}Classifier_block_{}_epoch_{}{}'.format(semi_string, conv_string, conv_block_num + 1,
                                                                num_epoch + epoch_offset, paper_string))
        else:
            fm.save_net(net, 'RotNet_rotation_{}{}'.format(num_epoch + epoch_offset, paper_string))
            fm.save_variable([loss_log, valid_accuracy_log, test_accuracy_log, max_accuracy, best_epoch],
                             'RotNet_rotation_{}{}'.format(num_epoch + epoch_offset, paper_string))

    return loss_log, valid_accuracy_log, test_accuracy_log, max_accuracy, best_epoch


def adaptive_learning(lr_list, epoch_change, momentum, weight_decay, net, criterion, trainloader, validloader=None,
                      testloader=None, classifier=None, conv_block_num=None, rot=None, use_paper_metric=False,
                      use_ConvClassifier=False, semi=None):
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
    :param criterion: the criterion to compute the loss
    :param trainloader: the training set wrapped by a loader
    :param validloader: the validation set wrapped by a loader
    :param testloader: the test set wrapped by a loader
    :param classifier: optional argument, if provided, the classifier will be attached to the feature map of the x-th
    convolutional block of the neural network (where x = conv_block_num) and trained for classification task
    :param conv_block_num: number of the RotNet convolutional block to which the classifier will be attached
    :param rot: list of classes for the rotation task. Possible classes are: '90', '180', '270'. Optional argument, if
    provided the neural network will be trained for the rotation task instead of the classification task.
    :param use_paper_metric: use the metric from the paper "Unsupervised Representation Learning by Predicting Image
    Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis. Default: False
    :param use_ConvClassifier: This parameter is not indented to be changed, but rather will be passed from the
    train_all_blocks function. In other words, you do not need to provide this argument if using this function.
    :param semi: This parameter is not indented to be changed, but rather will be passed from the train_semi function.
    In other words, you do not need to provide this argument if using this function.
    :return: loss_log: a list of all losses computed at each training epoch
             accuracy_log: a list of all validation accuracies computed at each training epoch
             max_accuracy: the highest accuracy achieved on the validation set so far
             best_epoch: the epoch in which the highest accuracy was achieved on the validation set
    """

    loss_log = []
    valid_accuracy_log = []
    test_accuracy_log = []
    max_accuracy = 0
    best_epoch = 0

    for i, lr in enumerate(lr_list):
        if i == 0:
            epoch_offset = 0
        else:
            epoch_offset = epoch_change[i-1]

        num_epoch = epoch_change[i] - epoch_offset

        if classifier is None:
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        else:
            optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                  nesterov=True)

        if i == (len(lr_list) - 1):
            printing = True
        else:
            printing = False

        tmp_loss_log, tmp_valid_accuracy_log, tmp_test_accuracy_log, max_accuracy, best_epoch = \
            train(num_epoch, net, criterion, optimizer, trainloader, validloader, testloader, classifier,
                  conv_block_num, epoch_offset, rot, printing, max_accuracy, best_epoch, use_paper_metric,
                  use_ConvClassifier, semi)

        loss_log += tmp_loss_log
        valid_accuracy_log += tmp_valid_accuracy_log
        test_accuracy_log += tmp_test_accuracy_log

    return loss_log, valid_accuracy_log, test_accuracy_log, max_accuracy, best_epoch


def train_all_blocks(conv_block_num, num_classes, lr_list, epoch_change, momentum, weight_decay, net, criterion,
                     trainloader, validloader=None, testloader=None, use_paper_metric=False, use_ConvClassifier=False,
                     optional_pooling=False):
    """
    Train classifiers on all convolutional blocks feature maps of a pre-trained RotNet.

    :param conv_block_num: number of convolutional blocks in the RotNet
    :param num_classes: number of classes in the classification task
    :param lr_list: a list of learning rates use for adaptive learning
    :param epoch_change: epochs where the learning rate should be change. Should have the same length as lr_list.
    :param momentum: momentum factor for stochastic gradient descent
    :param weight_decay: weight decay (L2 penalty) for stochastic gradient descent
    :param net: the pre-trained RotNet
    :param criterion: the criterion to compute the loss
    :param trainloader: the training set wrapped by a loader
    :param validloader: the validation set wrapped by a loader
    :param testloader: the test set wrapped by a loader
    :param use_paper_metric: use the metric from the paper "Unsupervised Representation Learning by Predicting Image
    Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis. Default: False
    :param use_ConvClassifier: If True, train convolutional block classifiers instead of a NonLinearClassifiers on the
    convolutional blocks feature maps of the RotNet. Default: False, in this case NonLinearClassifiers will be trained
    on the feature maps
    :param optional_pooling: If true, the classifiers are adjusted to fit the dimensions changed through applying an
    average pooling layer between the 3rd and 4th convolutional block
    :return: loss_log: a 2d list of all losses computed at each training epoch for each block
             accuracy_log: a 2d list of all validation accuracies computed at each training epoch for each block
             max_accuracy: list the highest accuracy achieved on the validation set so far
             best_epoch: list of the epoch in which the highest accuracy was achieved on the validation set
    """

    loss_log = []
    valid_accuracy_log = []
    test_accuracy_log = []
    max_accuracy = []
    best_epoch = []

    for i in range(conv_block_num):
        if i == 0:
            if use_ConvClassifier:
                clf = CC.ConvClassifier(num_classes, 96)
            else:
                clf = NLC.NonLinearClassifier(num_classes, 96*16*16)
        else:
            if use_ConvClassifier:
                clf = CC.ConvClassifier(num_classes, 192)
            else:
                if optional_pooling and i > 1:
                    clf = NLC.NonLinearClassifier(num_classes, 192*4*4)
                else:
                    clf = NLC.NonLinearClassifier(num_classes, 192*8*8)

        tmp_loss_log, tmp_valid_accuracy_log, tmp_test_accuracy_log, tmp_max_accuracy, tmp_best_epoch = \
            adaptive_learning(lr_list, epoch_change, momentum, weight_decay, net, criterion, trainloader, validloader,
                              testloader, clf, i+1, None, use_paper_metric, use_ConvClassifier)

        loss_log.append(tmp_loss_log)
        valid_accuracy_log.append(tmp_valid_accuracy_log)
        test_accuracy_log.append(tmp_test_accuracy_log)
        max_accuracy.append(tmp_max_accuracy)
        best_epoch.append(tmp_best_epoch)

    return loss_log, valid_accuracy_log, test_accuracy_log, max_accuracy, best_epoch


def train_semi(img_per_class, num_classes, trainset, testset, batch_size, semi_lr_lst, semi_epoch_change, super_lr_lst,
               super_epoch_change, momentum, weight_decay, net, criterion, use_paper_metric=False):
    """
    Run the semi-supervised learning experiment. As a benchmark the supervised NIN experiment will be performed with the
    same number of images per class.

    :param img_per_class: a list of numbers which represent the number of images per class used for training
    :param num_classes: number of classes in the classification task
    :param trainset: set of data used for training
    :param testset: set of data used for testing
    :param batch_size: size of the batch used during training
    :param semi_lr_lst:  a list of learning rates use for adaptive learning in the semi-supervised learning experiment
    :param semi_epoch_change: epochs where the learning rate should be change during the semi-supervised learning
    experiment. Should have the same length as semi_lr_list.
    :param super_lr_lst:  a list of learning rates use for adaptive learning in the supervised NIN experiment
    :param super_epoch_change: epochs where the learning rate should be change during the supervised NIN experiment.
    Should have the same length as super_lr_list.
    :param momentum: momentum factor for stochastic gradient descent
    :param weight_decay: weight decay (L2 penalty) for stochastic gradient descent
    :param net: the pre-trained RotNet from the Rotation Task
    :param criterion: the criterion to compute the loss
    :param use_paper_metric: use the metric from the paper "Unsupervised Representation Learning by Predicting Image
    Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis. Default: False
    :return: semi_loss_log, semi_accuracy_log, super_loss_log, super_accuracy_log
    """

    semi_loss_log = []
    semi_accuracy_log = []
    super_loss_log = []
    super_accuracy_log = []

    for num_img in img_per_class:
        trainloader, testloader = dl.make_dataloaders(trainset, testset, batch_size, subset=num_img)

        clf = CC.ConvClassifier(num_classes, 192)

        tmp_loss_log, _, tmp_test_accuracy_log, _, _ = adaptive_learning(semi_lr_lst, semi_epoch_change, momentum,
            weight_decay, net, criterion, trainloader, None, testloader, clf, 2, None, use_paper_metric, True, num_img)

        semi_loss_log.append(tmp_loss_log)
        semi_accuracy_log.append(tmp_test_accuracy_log)

        net = RN.RotNet(num_classes=10, num_conv_block=3, add_avg_pool=False)

        nin_tmp_loss_log, _, nin_tmp_test_accuracy_log, _, _ = adaptive_learning(super_lr_lst, super_epoch_change,
            momentum, weight_decay, net, criterion, trainloader, None, testloader, None, 2, use_paper_metric, False,
                num_img)

        super_loss_log.append(nin_tmp_loss_log)
        super_accuracy_log.append(nin_tmp_test_accuracy_log)

    return semi_loss_log, semi_accuracy_log, super_loss_log, super_accuracy_log

