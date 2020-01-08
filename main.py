"""
SAA - self adaptive algorithm
FAS - fast adaptive similarity filter
Images from https://pixabay.com/images/search/
"""
import time

import utils
import constants
import NEAVF

start = time.time()

# Read, resize and plot images
size = (300, 400)
img_list = utils.read_images(constants.DATASET_PATH)
img_list = utils.resize_img(img_list, size)
# utils.show_image_list(img_list, "Initial images!")

img_list_gauss = []
img_list_impulsive = []
index = 0
for img in img_list:
    img_list_gauss.append(utils.gaussian_noise(img, 0.1))
    img_list_impulsive.append(utils.salt_and_pepper(img, 0.1))
    index += 1
# utils.show_image_list(img_list_gauss, "Images with gaussian noise! 0.1")
# utils.show_image_list(img_list_impulsive, "Image with salt and pepper noise! 0.1")

# Two images with 2 values of noise
img_list_gauss_005 = []
img_list_gauss_015 = []
img_list_impulsive_005 = []
img_list_impulsive_015 = []
for i in range(2):
    # Make sure we have the same dimension of the list to compute the errors(SNR, MAE, PSNR)
    img_list.append(img_list[i])
    img_list_gauss_005.append(utils.gaussian_noise(img_list[i], 0.1))
    img_list_gauss_015.append(utils.gaussian_noise(img_list[i], 0.15))
    img_list_impulsive_005.append(utils.salt_and_pepper(img_list[i], 0.05))
    img_list_impulsive_015.append(utils.salt_and_pepper(img_list[i], 0.15))
# utils.show_image_list(img_list_gauss_005, "Images with gaussian noise! 0.05")
# utils.show_image_list(img_list_gauss_015, "Images with gaussian noise! 0.15")
# utils.show_image_list(img_list_impulsive_005, "Image with salt and pepper noise! 0.05")
# utils.show_image_list(img_list_impulsive_015, "Image with salt and pepper noise! 0.15")


img_list_impulsive = img_list_impulsive + img_list_impulsive_005 + img_list_impulsive_015
img_list_impulsive_median = []
img_list_impulsive_neavf = []
index = 0
for img in img_list_impulsive:
    img_list_impulsive_median.append(utils.median_filter(img_list_impulsive[index]))
    img_list_impulsive_neavf.append(NEAVF.NEAVF(img_list_impulsive[index]))
    index += 1
# utils.show_image_list(img_list_impulsive_median, "Image with salt and pepper after median filter")
# utils.show_image_list(img_list_impulsive_neavf, "Image with salt and pepper after NEAVF filter")


img_list_gauss = img_list_gauss + img_list_gauss_005 + img_list_gauss_015
img_list_gauss_mean = []
img_list_gauss_neavf = []
index = 0
for img in img_list_gauss:
    img_list_gauss_mean.append(utils.mean_filter(img_list_gauss[index]))
    img_list_gauss_neavf.append(NEAVF.NEAVF(img_list_gauss[index]))
    index += 1
# utils.show_image_list(img_list_gauss_mean, "Image with gauss after mean filter!")
# utils.show_image_list(img_list_gauss_neavf, "Image with gauss after NEAVF filter!")

# Noise measure
psnr_gauss = []
psnr_impulsive = []
psnr_NEAVF_impulsive = []
psnr_NEAVF_gauss = []
mae_gauss = []
mae_impulsive = []
mae_NEAVF_impulsive = []
mae_NEAVF_gauss = []
img_list.append(img_list[0])
img_list.append(img_list[1])
for i in range(len(img_list)):
    psnr_gauss.append(utils.PSNR(img_list[i], img_list_gauss_mean[i]))
    psnr_impulsive.append(utils.PSNR(img_list[i], img_list_impulsive_median[i]))
    psnr_NEAVF_impulsive.append(utils.PSNR(img_list[i], img_list_impulsive_neavf[i]))
    psnr_NEAVF_gauss.append(utils.PSNR(img_list[i], img_list_gauss_neavf[i]))
    print("PSNR for Image {0} after gaussian noise + filter(median) = {1};".format(i, psnr_gauss[i]))
    print("PSNR for Image {0} after impulsive noise + filter(median) = {1};".format(i, psnr_impulsive[i]))
    print("PSNR for Image {0} after impulsive noise + filter(NEAVF) = {1};".format(i, psnr_NEAVF_impulsive[i]))
    print("PSNR for Image {0} after gauss noise + filter(NEAVF) = {1};".format(i, psnr_NEAVF_gauss[i]))
    mae_gauss.append(utils.MAE(img_list[i], img_list_gauss[i]))
    mae_impulsive.append(utils.MAE(img_list[i], img_list_impulsive[i]))
    mae_NEAVF_impulsive.append(utils.MAE(img_list[i], img_list_impulsive_neavf[i]))
    mae_NEAVF_gauss.append(utils.MAE(img_list[i], img_list_gauss_neavf[i]))
    print("MAE for Image {0} after gaussian noise + filter(mean) = {1};".format(i, mae_gauss[i]))
    print("MAE for Image {0} after impulsive noise + filter(median) = {1};".format(i, mae_impulsive[i]))
    print("MAE for Image {0} after impulsive noise + filter(NEAVF) = {1};".format(i, mae_NEAVF_impulsive[i]))
    print("MAE for Image {0} after gaussian noise + filter(NEAVF) = {1};".format(i, mae_NEAVF_gauss[i]))

end = time.time()
print("Total time = {0} seconds.".format(end - start))
