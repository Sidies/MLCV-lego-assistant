import cv2
import albumentations as A
import shutil
import numpy as np
import pandas as pd
import csv
import xml.etree.ElementTree as et
import os


transform = A.Compose([
    A.Resize(height=256, width=256),  # Größeres Resize auf 256x256
    A.RandomCrop(height=224, width=224),  # Random Crop auf 224x224
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.GaussianBlur(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
], bbox_params=A.BboxParams(format="pascal_voc"))





# Get bbox via xml filename from Annotations folder
def get_bbox(filename: str):
    filename += '.xml'
    path_annotations ='../data/lego/Annotations'
    for file in os.listdir(path=path_annotations):
        if file.endswith(".xml") and file == filename:
            path_file = os.path.join(path_annotations, file)
            tree = et.parse(path_file)
            root = tree.getroot()

            xmin = float(root.find("./object/bndbox/xmin").text)
            ymin = float(root.find("./object/bndbox/ymin").text)
            xmax = float(root.find("./object/bndbox/xmax").text)
            ymax = float(root.find("./object/bndbox/ymax").text)
            label = root.find("./object/name").text
            return [[xmin, ymin, xmax, ymax, label]]

# Return an adapted unique filename for entered filename        
def get_unique_filename(filename: str):
    # Check if the base file already exists
    filename += '.xml'
    path = "../data/lego/Annotations/" + filename
    if not os.path.exists(path):
        return filename
    # Search for a unique number for the name
    filename_split = filename.split(".")
    base_name = filename_split[0]
    extension = filename_split[1]
    counter = 1
    unique_filename = base_name + "_" + str(counter) + '.' + extension
    path = "../data/lego/Annotations/" + unique_filename
    while os.path.exists(path):
        counter += 1
        unique_filename = base_name + "_" + str(counter) + '.' + extension
        path = "../data/lego/Annotations/" + unique_filename
        
    return unique_filename

# Creates a bbox-xml-file for new augmented file
def create_xml_file(initial_filename: str, new_filename: str, transformed_bbox):
    initial_filename += '.xml'
    path_annotations ='../data/lego/Annotations/'
    for xml_file in os.listdir(path_annotations):
        if xml_file == initial_filename:
            path_file = os.path.join(path_annotations, xml_file)
            new_filename = new_filename.split(".")[0] + ".xml"
            tree = et.parse(path_file)
            root = tree.getroot()

            filename_element = root.find("filename")
            filename_element.text = new_filename
            xmin = root.find("./object/bndbox/xmin")
            xmin.text = str(transformed_bbox[0][0])
            ymin = root.find("./object/bndbox/ymin")
            ymin.text = str(transformed_bbox[0][1])
            xmax = root.find("./object/bndbox/xmax")
            xmax.text = str(transformed_bbox[0][2])
            ymax = root.find("./object/bndbox/ymax")
            ymax.text = str(transformed_bbox[0][3])
            width = root.find('./size/width')
            width.text = '224'
            height = root.find('./size/height')
            height.text = '224'
            
            new_path_file = os.path.join(path_annotations, new_filename)
            tree.write(new_path_file)
            
def write_name_to_file(image_name: str, path: str):
    with open(path, "a") as file:
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
        
def write_train_trainval(foldername:str):
    traintxt_path = '../data/lego/ImageSets/Main/train.txt'
    trainvaltxt_path = '../data/lego/ImageSets/Main/trainval.txt'
    base_path = '../data/raw/' 
    target_path = os.path.join(base_path, foldername)
    folder = os.listdir(target_path)
    for filename in folder:
        write_name_to_file(image_name=filename, path=traintxt_path)
        write_name_to_file(image_name=filename, path=trainvaltxt_path)
        
def write_test(foldername:str):
    testtxt_path = '../data/lego/ImageSets/Main/test.txt'
    base_path = '../data/raw/' 
    target_path = os.path.join(base_path, foldername)
    folder = os.listdir(target_path)
    for filename in folder:
        write_name_to_file(image_name=filename, path=testtxt_path)
        
def write_val(foldername:str):
    valtxt_path = '../data/lego/ImageSets/Main/val.txt'
    base_path = '../data/raw/' 
    target_path = os.path.join(base_path, foldername)
    folder = os.listdir(target_path)
    for filename in folder:
        write_name_to_file(image_name=filename, path=valtxt_path)
        
def reset():
    delete_all_entries('../data/lego/ImageSets/Main/train.txt')
    delete_all_entries('../data/lego/ImageSets/Main/trainval.txt')
    delete_all_entries('../data/lego/ImageSets/Main/test.txt')
    delete_all_entries('../data/lego/ImageSets/Main/val.txt')
    delete_all_files('../data/lego/JPEGImages/')
    delete_all_files('../data/lego/Annotations/')
    copy_all_files(source_folder='../data/labeling/', destination_folder='../data/lego/Annotations/')
    #write_traintxt("Iteration_1")
    
def augmentation(folder_name: str):
    path = "../data/raw/" + folder_name
    for file in os.listdir(path):
        file_path = os.path.join(path, file) # Build path to file
        image = cv2.imread(file_path) # Get image via path
        file = file.split('.')[0]
        bboxes = get_bbox(file) # Get bbox of file
        
        # Transform img and its bbox
        transformed = transform(image=image, bboxes=bboxes) # Augmentation on image and bbox
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        
        new_filename_xml = get_unique_filename(file) # Create a unique filename for new aug file
        create_xml_file(initial_filename=file, new_filename=new_filename_xml, transformed_bbox=transformed_bboxes) # Create XML-file for new aug file
        
        # Save aug img in JPEGImages
        path_write = '../data/lego/JPEGImages/' 
        new_filename_jpg = new_filename_xml.split('.')[0] + '.jpg'
        cv2.imwrite(
            path_write + new_filename_jpg,
            transformed_image,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        write_name_to_file(image_name=new_filename_jpg, path='../data/lego/ImageSets/Main/train.txt')
        write_name_to_file(image_name=new_filename_jpg, path='../data/lego/ImageSets/Main/trainval.txt')
        
        # how aug img + bbox
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
