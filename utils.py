import glob
import cv2
import numpy as np
from skimage.transform import resize
import math
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

def read_images(path):
    """
    :param path: Path to the image dataset.
    :return: A list of images.
    """
    files = []
    extension = ['jpg', 'JPG']
    [files.extend(glob.glob(path + "/*." + e)) for e in extension]
    img_list = [cv2.imread(file) for file in files]
    return img_list

def resize_img(img_list, size):
    """
    :param img_list: List with images.
    :param size: Wanted size.
    :return: Resized image.
    """
    index = 0
    for img in img_list:
        img_list[index] = resize(img, size).astype(np.float32)
        index += 1
    return img_list

def show_image_list(img_list, figure_name):
    """
    :param img_list: List of images.
    :param figure_name: Name of the final figure.
    :return: All the images in one figure.
    """
    index = 1
    img_size = len(img_list)
    plot_dim = math.ceil(math.sqrt(img_size))
    plt.figure(figure_name, figsize=(15, 15), dpi=100)
    for img in img_list:
        plt.subplot(plot_dim, plot_dim, index), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Image {0}".format(index))
        index += 1
    plt.show()
    plt.close('all')

def kernel_c(img, window_size, i, j, channel):
    """
    Select from an image a window centered at (i, j), with the dimension of window_size.
    :param img: Source image.
    :param window_size: The size of window(window_size*window_size matrix).
    :param i: Horizontal coordinate of the center.
    :param j: Vertical coordinate of the center.
    :param channel: If the image is RGB, select the channel.
    :return: The centered window.
    """
    dim = window_size // 2
    return img[i - dim:i + dim + 1, j - dim:j + dim + 1, channel]

def kernel(img, window_size, i, j):
    """
    Select from an image a window centered at (i, j), with the dimension of window_size.
    :param img: Source image.
    :param window_size: The size of window(window_size*window_size matrix).
    :param i: Horizontal coordinate of the center.
    :param j: Vertical coordinate of the center.
    :return: The centered window.
    """
    dim = window_size // 2
    return img[i - dim:i + dim + 1, j - dim:j + dim + 1]

def gaussian_noise(image, sigma):
    """
    :param image: Source image.
    :param sigma: Measure of the amount of variation or dispersion of a pixel.
    :return: np.array: Image with gaussian noise.
    """
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 1)
    return np.float32(noisy_image)

def salt_and_pepper(image, total_area):
    """
    :param image: Initial image.
    :param total_area: Area affected by noise.
    :return: np.array Image with salt an pepper noise.
    """
    amount = total_area / 2
    row, col, ch = image.shape
    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out

def median_filter(image):
    """
    :param image: The input image.
    :return: numpy array: Image without salt and pepper noise.
    """
    row, col, ch = image.shape
    out_image = image.astype(np.float32).copy()
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            R = np.sort(kernel_c(image, 3, i, j, 0), axis=None)
            G = np.sort(kernel_c(image, 3, i, j, 1), axis=None)
            B = np.sort(kernel_c(image, 3, i, j, 2), axis=None)
            out_image[i, j, 0] = R[4]
            out_image[i, j, 1] = G[4]
            out_image[i, j, 2] = B[4]
    return out_image

def mean_filter(image):
    """
    :param image: The input image.
    :return: numpy array: Image without gaussian noise.
    """
    row, col, ch = image.shape
    out_image = image.astype(np.float32).copy()
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            out_image[i, j, 0] = np.sum(kernel_c(image, 3, i, j, 0), axis=None) / 9
            out_image[i, j, 1] = np.sum(kernel_c(image, 3, i, j, 1), axis=None) / 9
            out_image[i, j, 2] = np.sum(kernel_c(image, 3, i, j, 2), axis=None) / 9
    return out_image

def MSE(initial_image, final_image):
    """
    Mean Squared Error(MSE).
    :param initial_image: Image without noise.
    :param final_image: Image affected by noise.
    :return: Value of mean squared error.
    """
    return (np.square(initial_image - final_image)).mean(axis=None)

def PSNR(initial_image, final_image):
    """
    Peak Signal to Noise Ratio
    :param initial_image: Image without noise.
    :param final_image: Image affected by noise.
    :return:
    """
    mse = MSE(initial_image, final_image)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def MAE(initial_image, final_image):
    """
    Mean Absolute Error is the average vertical distance between each point and the identity line.
    :param initial_image: Image without noise.
    :param final_image: Image affected by noise.
    :return: Value of mean absolute value.
    """
    row, col, ch = initial_image.shape
    dimension = row * col * ch
    return (np.sum(np.abs(initial_image - final_image))) / dimension
