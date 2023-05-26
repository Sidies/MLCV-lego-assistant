import cv2
import numpy as np
import os
import json
import albumentations as A


transform = A.Compose(
    [
        A.Resize(height=480, width=640),
        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="coco"),
)


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


def transformed(path, json_file_path, count):
    trans_img_bbox = []
    if os.path.isdir(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            for obj in os.listdir(folder_path):
                obj_path = os.path.join(folder_path, obj)
                img = cv2.imread(obj_path)
                bbox = get_bbox_from_json(json_file_path, obj)
                for i in range(0, count):
                    transformed = transform(image=img, bboxes=bbox)
                    trans_img_bbox.append(transformed)
                    return trans_img_bbox
