import torch


def rot90(image):
    '''
    Rotate an image by 90 degree by first transposing the image and then flipping it vertically

    :param image: The image that should be rotated.
    :return: The rotated image.
    '''

    # transpose
    trans_im = image.permute(0, 2, 1)

    # flip vertically
    inv_idx = torch.range(trans_im.size(2) - 1, 0, -1).long()
    rot_im = trans_im.index_select(2, inv_idx)

    return  rot_im

