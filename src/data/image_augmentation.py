import cv2
import numpy as np
import os
import json


def get_bbox_from_json(json_file_path, file_name):
    if json_file_path is None:
        print("JSON-Datei nicht gefunden.")
        return

    with open(json_file_path, "r") as file:
        json_data = json.load(file)

        for key in json_data["_via_img_metadata"].keys():
            if file_name in str(key):
                filename_in_json = json_data["_via_img_metadata"][str(key)]["filename"]
                if filename_in_json == file_name:
                    region_data = json_data["_via_img_metadata"][str(key)]["regions"][
                        0
                    ]["shape_attributes"]
                    x = region_data["x"]
                    y = region_data["y"]
                    width = region_data["width"]
                    height = region_data["height"]
                    # Get Key of "region_attributes"
                    region_attributes = list(
                        json_data["_via_img_metadata"][str(key)]["regions"][0][
                            "region_attributes"
                        ].keys()
                    )[0]
                    label = json_data["_via_img_metadata"][str(key)]["regions"][0][
                        "region_attributes"
                    ][region_attributes]

                    # coco [x_min, y_min, width, height]
                    bboxes = [[x, y, width, height, label]]
                    return bboxes


def plot_img_with_bbox(transformed_image, transformed_bboxes):
    x, y, width, height, label = transformed_bboxes[0]
    cv2.rectangle(
        transformed_image,
        (int(x), int(y)),
        (int(x + width), int(y + height)),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        transformed_image,
        label,
        (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Image with Bounding Box", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(image, angle):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def scale_image(image, fx, fy):
    return cv2.resize(image, None, fx=fx, fy=fy)


def translate_image(image, x, y):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (cols, rows))


def flip_image(image, direction):
    return cv2.flip(image, direction)
