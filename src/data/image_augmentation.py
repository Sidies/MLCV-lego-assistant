import cv2
import albumentations as A
import numpy as np
import pandas as pd
import csv
import xml.etree.ElementTree as et
import os


transform = A.Compose(
    [
        A.Resize(height=480, width=640),
        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc"),
)


def create_annotation_file():
    csv_path = "../data/labeling/annotations_bbox.csv"
    path_labels = "../data/labeling"
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])

        for filename in os.listdir(path_labels):
            if filename.endswith(".xml"):
                path_file = os.path.join(path_labels, filename)
                tree = et.parse(path_file)
                root = tree.getroot()

                filename = root.find("filename").text

                label = root.find("./object/name").text
                xmin = root.find("./object/bndbox/xmin").text
                xmax = root.find("./object/bndbox/xmax").text
                ymin = root.find("./object/bndbox/ymin").text
                ymax = root.find("./object/bndbox/ymax").text

                writer.writerow([filename, label, xmin, xmax, ymin, ymax])


def get_bbox(filename):
    csv_file = "../data/labeling/annotations_bbox.csv"
    df = pd.read_csv(csv_file)
    filtered = df[df["ImageID"] == filename]
    labelname = filtered["LabelName"].iloc[0]
    xmin = filtered["XMin"].iloc[0]
    xmax = filtered["XMax"].iloc[0]
    ymin = filtered["YMin"].iloc[0]
    ymax = filtered["YMax"].iloc[0]
    return [[xmin, xmax, ymin, ymax, labelname]]


def write_bbox_to_csv(transformed_bbox, image_name):
    # Pascal Voc [x_min, y_min, x_max, y_max, label]
    xmin = transformed_bbox[0][0]
    ymin = transformed_bbox[0][1]
    xmax = transformed_bbox[0][2]
    ymax = transformed_bbox[0][3]
    label = transformed_bbox[0][4]

    csv_file = "../data/labeling/annotations_bbox.csv"
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([image_name, label, xmin, xmax, ymin, ymax])


def enlarge_dataset(factor: int):
    path_img = "../data/raw/"
    for filename in os.listdir(path_img):
        path_file = os.path.join(path_img, filename)
        image = cv2.imread(path_file)
        bboxes = get_bbox(filename)
        bboxes[0][1], bboxes[0][2] = bboxes[0][2], bboxes[0][1]

        for i in range(0, factor):
            transformed = transform(image=image, bboxes=bboxes)
            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            filename_split = filename.split(".")
            filename = filename_split[0] + "_P." + filename_split[1]
            cv2.imwrite(
                "../data/processed/" + filename,
                transformed_image,
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )
            write_bbox_to_csv(transformed_bboxes, filename)
            # plot_img_with_bbox(transformed_image, transformed_bboxes)


def remove_processed_imgs():
    path = "../data/processed/"
    for file_name in os.listdir(path):
        file = path + file_name
        if os.path.isfile(file):
            os.remove(file)


def plot_img_with_bbox(transformed_image, transformed_bboxes):
    x = transformed_bboxes[0][0]
    y = transformed_bboxes[0][1]
    width = transformed_bboxes[0][2] - transformed_bboxes[0][0]
    height = transformed_bboxes[0][3] - transformed_bboxes[0][1]
    label = transformed_bboxes[0][4]
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
