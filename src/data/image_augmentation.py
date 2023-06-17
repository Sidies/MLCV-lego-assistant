import cv2
import albumentations as A
import shutil
import numpy as np
import pandas as pd
import csv
import xml.etree.ElementTree as et
import os


transform = A.Compose(
    [
        A.Resize(height=224, width=224),
        #A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc"),
)

def get_bbox(xml_filename: str):
    path_annotations ='../data/lego/Annotations'
    for file in os.listdir(path=path_annotations):
        if file.endswith(".xml") and file == xml_filename:
            path_file = os.path.join(path_annotations, file)
            tree = et.parse(path_file)
            root = tree.getroot()

            filename = root.find("filename").text

            xmin = float(root.find("./object/bndbox/xmin").text)
            ymin = float(root.find("./object/bndbox/ymin").text)
            xmax = float(root.find("./object/bndbox/xmax").text)
            ymax = float(root.find("./object/bndbox/ymax").text)

            label = root.find("./object/name").text
            # Pascal Voc: [x_min, y_min, x_max, y_max, label]
            return [[xmin, ymin, xmax, ymax, label]]
        
def get_unique_filename(filename: str):
    # Überprüfe, ob die Basisdatei bereits existiert
    path = "../data/lego/Annotations/" + filename
    if not os.path.exists(path):
        return filename
    # Suche nach einer eindeutigen Nummer für den Namen
    filename_split = filename.split(".")
    base_name = filename_split[0]
    extension = filename_split[1]
    counter = 1
    unique_name = base_name + "_" + str(counter) + '.' + extension
    while os.path.exists(unique_name):
        counter += 1
        unique_name = base_name + "_" + str(counter) + '.' + extension

    return unique_name

def create_bbox_xml_file(filename: str, filename_new: str):
    bbox = get_bbox(filename)
    # Pascal Voc: [x_min, y_min, x_max, y_max, label]
    path_annotations ='../data/lego/Annotations/'
    for xml_file in os.listdir(path_annotations):
        if xml_file == filename:
            path_file = os.path.join(path_annotations, xml_file)
            new_filename = filename_new.split(".")[0] + ".xml"
            tree = et.parse(path_file)
            root = tree.getroot()

            filename_element = root.find("filename")
            filename_element.text = new_filename
            xmin = root.find("./object/bndbox/xmin")
            xmin.text = str(bbox[0][0])
            ymin = root.find("./object/bndbox/ymin")
            ymin.text = str(bbox[0][1])
            xmax = root.find("./object/bndbox/xmax")
            xmax.text = str(bbox[0][2])
            ymax = root.find("./object/bndbox/ymax")
            ymax.text = str(bbox[0][3])
            label = root.find("./object/name")
            
            new_path_file = os.path.join(path_annotations, new_filename)
            tree.write(new_path_file)
            
def write_img_name(image_name: str, txt_path: str):
    with open(txt_path, "a") as file:
        file.write(image_name.split('.')[0] + "\n")
        
def delete_all_entries(file_path: str):
    with open(file_path, 'w') as file:
        file.truncate(0)
        
def delete_all_files(folder_path: str):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
def copy_all_files(source_folder: str, destination_folder: str):
    file_list = os.listdir(source_folder)
    for file_name in file_list:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy(source_path, destination_path)
        
def reset():
    delete_all_entries('../data/lego/ImageSets/Main/train.txt')
    delete_all_files('../data/lego/JPEGImages/')
    delete_all_files('../data/lego/Annotations/')
    copy_all_files(source_folder='../data/labeling/', destination_folder='../data/lego/Annotations/')
    
def augmentation(folder_name: str):
    path = "../data/raw/" + folder_name
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        image = cv2.imread(file_path)
        xml_filename = file.split('.')[0] + '.xml'
        bboxes = get_bbox(xml_filename)
        
        # transform img and its bbox
        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        
        unique_filename_xml = get_unique_filename(xml_filename)
        print('augmentation filename_new:', unique_filename_xml)
        # create bbox file for aug img
        create_bbox_xml_file(xml_filename, unique_filename_xml)
        
        # save aug img in JPEGImages
        path_write = '../data/lego/JPEGImages/' 
        unique_filename_jpg = unique_filename_xml.split('.')[0] + '.jpg'
        cv2.imwrite(
            path_write + unique_filename_jpg,
            transformed_image,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        write_img_name(unique_filename_jpg, '../data/lego/ImageSets/Main/train.txt')
        #plot_img_with_bbox(transformed_image, transformed_bboxes)


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
