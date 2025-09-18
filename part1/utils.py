import cv2
import numpy as np

def save_concatenated_images(original_img, cam_image, save_path):
    # Convert image formats if necessary
    original_img = (original_img * 255).astype(np.uint8)
    concatenated = cv2.hconcat([original_img, cam_image])
    cv2.imwrite(save_path, concatenated)
