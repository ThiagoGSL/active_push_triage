import cv2 as cv
import numpy as np

def rgbImg_to_binaryImg(rgb_img, min_max_hue, min_max_saturation, min_max_val=np.array([0,255])):
    img_hsv = cv.cvtColor(rgb_img, cv.COLOR_RGB2HSV) # RGB -> HSV
    hue = img_hsv[:,:,0]
    sat = img_hsv[:,:,1]
    val = img_hsv[:,:,2]

    min_hue, max_hue = min_max_hue
    min_sat, max_sat = min_max_saturation
    min_val, max_val = min_max_val
    mask = np.bitwise_and(
                np.bitwise_and(val>=min_val, val <= max_val), 
                np.bitwise_and(np.bitwise_and(sat >= min_sat, sat <= max_sat), np.bitwise_and(hue >= min_hue, hue <= max_hue)))

    binary_img = np.zeros(hue.shape)
    binary_img[mask] = 1

    return binary_img