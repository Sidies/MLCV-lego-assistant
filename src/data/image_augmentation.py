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

        for key, val in json_data.items():
            if file_name in key:
                filename_in_json = json_data[key]["filename"]
                if filename_in_json == file_name:
                    region_data = json_data[key]["regions"][0]["shape_attributes"]
                    x = region_data["x"]
                    y = region_data["y"]
                    width = region_data["width"]
                    height = region_data["height"]
                    # Get Key of "region_attributes"
                    label = json_data[key]["regions"][0]["region_attributes"]["label"]

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


def transformed(img_bbox, transform):
    trans_img_bbox = []
    path = img_bbox["path_image"]
    json_file_path = img_bbox["path_json"]
    if os.path.isdir(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            for obj in os.listdir(folder_path):
                obj_path = os.path.join(folder_path, obj)
                img = cv2.imread(obj_path)
                bbox = get_bbox_from_json(json_file_path, obj)
                # for i in range(0, COUNT):
                transformed = transform(image=img, bboxes=bbox)
                trans_img_bbox.append(transformed)
    return trans_img_bbox
