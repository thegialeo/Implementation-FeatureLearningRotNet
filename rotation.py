import torch


def rot90(image):
    '''
    Rotate an image by 90 degree by first transposing the image and then flipping it vertically.

    :param image: The image that should be rotated.
    :return: The rotated image.
    '''

    # transpose
    trans_im = image.permute(0, 2, 1)

    # flip vertically
    flip_idx = torch.range(trans_im.size(2) - 1, 0, -1).long()
    rot_im = trans_im.index_select(2, flip_idx)

    return  rot_im


def rot180(image):
    '''
    Rotate an image by 180 degree by first flipping the image vertically and then horizontally.

    :param image: The image that should be rotated.
    :return: The rotated image.
    '''

    # flip vertically
    vert_idx = torch.range(image.size(2) - 1, 0, -1).long()
    vert_im = image.index_select(2, vert_idx)

    # flip horizontally
    hor_idx = torch.range(vert_im.size(1) - 1, 0, -1).long()
    rot_im = vert_im.index_select(1, hor_idx)

    return rot_im


def rot270(image):
    '''
    Rotate an image by 270 degree by first flipping the image vertically and then transposing it.

    :param image: The image that should be rotated.
    :return: The rotated image
    '''

    # flip vertically
    vert_idx = torch.range(image.size(2) - 1, 0, -1).long()
    vert_im = image.index_select(2, vert_idx)

    # transpose
    rot_im = vert_im.permute(0, 2, 1)

    return rot_im


def apply(func, M):
    '''
    Applies a function on valid arguments or list of arguments.

    :param func: The function that we want to apply.
    :param M: The argument or list of arguments that will be passed to the function.
    :return: The result that the function returns.
    '''

    tList = [func(m) for m in torch.unbind(M, dim=0)]
    res = torch.stack(tList, dim=0)

    return res


def create_rot_batch(images, labels, rot=['90', '180', '270']):
    '''
    Takes a mini-batch of images with the corresponding labels and adds rotated versions of the images to the
    mini-batch. The original labels are modified accordingly to fit the newly create mini-batch. Furthermore, the
    corresponding rotation labels are created for the new mini-batch. By default all 3 rotations (90, 180 and
    270 degree) will be added, leading to a 4 times bigger new mini-batch.

    :param images: images of a mini-batch to rotate
    :param labels: labels corresponding to the images
    :param rot: list of rotations that should be added to the mini-batch. Possible rotations are: '90', '180' and '270'
    :return: rot_batch: original mini-batch with the rotated images added, class_labels: corresponding labels for the
    classification task, rot_labels: corresponding labels for the rotation task

    '''

    return rot_batch, class_labels, rot_labels


