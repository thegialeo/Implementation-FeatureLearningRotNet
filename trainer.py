import torch
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
            best_epoch = epoch + 1
            max_accuracy = accuracy
            if rot is None:
                if classifier is None:
                    torch.save(net.state_dict(), 'models/RotNet_classification_{}'.format(best_epoch))
                else:
                    torch.save(classifier.state_dict(), 'models/classifier_{}'.format(best_epoch))
            else:
                torch.save(net.state_dict(), 'models/RotNet_rotation_{}'.format(best_epoch))

    # printing
    if printing:
        print('highest validation accuracy: {:.3f} was achieved at epoch: {}'.format(max_accuracy, best_epoch))
        print('Finished Training')

    return loss_log, accuracy_log, max_accuracy, best_epoch


def adaptive_learning(num_epoch, lr, epoch_change, momentum, weight_decay, net, trainloader, validloader, criterion, optimizer,
                      classifier=None, conv_block_num=None, rot=None, use_paper_metric=False):
    """
    Use adaptive learning rate to train the neural network.

    Optional: If rot is provided, the neural network is trained for rotation prediction task instead of the
    classification task (neural network is used for training).

    Optional: If a classifier and conv_block_num is provided, the classifier is attached to the feature map of
    the x-th convolutional block of the neural network (where x = conv_block_num) and trained for the classification
    task. Only the classifier is trained, not the neural network itself. No fine tuning.

    :param num_epoch: number of training epochs
    :param lr: a list of learning rates use for adaptive learning
    :param epoch_change: epochs where the learning rate should be change to the next rate from the list 'lr'
    :param momentum: 
    :param weight_decay:
    :param net: neural network that should be trained
    :param trainloader:
    :param validloader:
    :param criterion:
    :param optimizer:
    :param classifier:
    :param conv_block_num:
    :param rot:
    :param use_paper_metric:
    :return:
    """
