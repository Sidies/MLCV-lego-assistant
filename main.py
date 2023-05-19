import os
import cv2
import pandas as pd
from src.data import image_augmentation
from glob import glob

def start_augmentation():
    # start image augmentation
    input_dir = "data/raw*"
    output_dir = "output_folder_path/"

    for file in glob(input_dir):
        # Read image
        img = cv2.imread(file)

        # Apply different augmentation techniques
        rotated_img = image_augmentation.rotate_image(img, 30)
        scaled_img = image_augmentation.scale_image(img, 0.5, 0.5)
        translated_img = image_augmentation.translate_image(img, 50, 50)
        flipped_img = image_augmentation.flip_image(img, 1)  # 0 for vertical flip, 1 for horizontal flip

        # Save the augmented images
        cv2.imwrite(output_dir + "rotated_" + os.path.basename(file), rotated_img)
        cv2.imwrite(output_dir + "scaled_" + os.path.basename(file), scaled_img)
        cv2.imwrite(output_dir + "translated_" + os.path.basename(file), translated_img)
        cv2.imwrite(output_dir + "flipped_" + os.path.basename(file), flipped_img)
