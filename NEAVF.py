import numpy as np
import time

def AVD(kernel, k_size):
    """
    Aggregated Vector Distance(AVD)
    :param kernel: Window of pixels.
    :param k_size: Numbers of aggregated vector distance(number of pixels).
    :return: Computed aggregated vector distance.
    """
    R = np.zeros(9)
    for i in range(0, k_size):
        for j in range(1, k_size):
            R[i] += np.sqrt((kernel[i][0] - kernel[j][0])**2 + (kernel[i][1] - kernel[j][1])**2 + (kernel[i][2] - kernel[j][2])**2)
    return R

def DALS(kernel, index):
    """
    :param kernel: Here we select only the third index which represents R value.
    :param index: The order of the central pixel.
    :return: Distribution adapted local similarity.
    """
    lambda_used = 1
    F = ((1 / 4) * (kernel[4][3] - kernel[0][3]) + lambda_used) / ((1 / index) * (kernel[index][3] - kernel[0][3]) + 1)
    return F

def NEAVF(image):
    """
    :param image: Image with noise.
    :return: np.array Filtrated image.
    """
    gamma = 0.5
    delta = 9

    start = time.time()
    row, col, _ = image.shape
    img_out = image.astype(np.float32).copy()
    img_result = image.astype(np.float32).copy()
    F = np.ones((row, col))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            kernel = np.zeros((9, 4))
            kernel[:, 0:3] = image[i - 1:i + 2, j - 1:j + 2, :].reshape((9, 3))
            kernel[[0, 4]] = kernel[[4, 0]]
            CP = kernel[0]
            kernel[:, 3] = AVD(kernel, 9)
            kernel = kernel[kernel[:, 3].argsort(kind='mergesort')]
            img_result[i][j][0] = kernel[0][0]
            img_result[i][j][1] = kernel[0][1]
            img_result[i][j][2] = kernel[0][2]
            for k in range(9):
                if CP[0] == kernel[k][0] and CP[1] == kernel[k][1] and CP[2] == kernel[k][2]:
                    l = k
                    break
            if l >= 4:
                F[i][j] = DALS(kernel, l)
            else:
                F[i][j] = 1

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            H = np.sum(np.power(F[i - 1:i + 2, j - 1:j + 2].reshape(9), gamma))
            if H >= delta:
                img_out[i][j][:] = image[i][j][:]
            else:
                img_out[i][j][:] = img_result[i][j][:]

    stop = time.time()
    print("Time NEAVF = {0}".format(stop-start))
    return img_out
