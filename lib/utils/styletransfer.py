import cv2
import numpy as np
import os
import utils.vis as vis_utils

DIRECTORY_TO_WATCH = "/mnt/api_files/input/"
DIRECTORY_TEMP = "/mnt/api_files/tmp/"
DIRECTORY_TO_WRITE = "/mnt/api_files/output/"


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def preProcessing(gammaImg):
  # adjusting gamma to brighten the image
  gammaImg = adjust_gamma(gammaImg, 1.0)

  # Guassian blur and adding weights to ensure beautification.
  gaussian_3 = cv2.GaussianBlur(gammaImg, (9,9), 10.0)
  gammaImg = cv2.addWeighted(gammaImg, 1.4, gaussian_3, 0.0, 0, gammaImg)

  # As the name indicates...Denoising
  # Look into reducing the redness in the result image
  gammaImg = cv2.fastNlMeansDenoisingColored(gammaImg,None,10,10,7,21)
  return gammaImg


