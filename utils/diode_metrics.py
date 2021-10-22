import numpy as np


def angular_error(gt, pred):

    # compute the vector product between gt and prediction for each pixel
    angularDist = (gt * pred).sum(axis=-1)

    # compute the angle from vector product
    angularDist = np.arccos(np.clip(angularDist, -1.0, 1.0))

    # convert radius to degrees
    angularDist = angularDist / np.pi * 180

    # find mask
    mask = np.float32(np.sum(gt ** 2, axis=-1) > 0.9)
    mask = mask != 0.0

    # only compute pixels under mask
    angularDist_masked = angularDist[mask]

    return angularDist_masked
