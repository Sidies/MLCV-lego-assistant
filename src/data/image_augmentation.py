import cv2
import numpy as np
import os
#import xmltodict
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import pandas as pd

voc_pascal_access = {
    ''
}

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


def xml_to_df(path):
    xml_list = []
       
    for xml_file in os.listdir(path):
        '''one file corresponds to one picture'''
        tree = ET.parse(path+'/'+xml_file)
        root = tree.getroot()
        
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = float(bbx.find('xmin').text)
            ymin = float(bbx.find('ymin').text)
            xmax = float(bbx.find('xmax').text)
            ymax = float(bbx.find('ymax').text)
            label = member.find('name').text

            value = (root.find('filename').text, # image ID
                     'freeform', #Source
                     label, #Labelname
                     1, # Confidence
                     xmin,
                     xmax,
                     ymin,
                     ymax) #,
                     #0, #bunch of booleans, we can also try other values
                     #0,
                     #0,
                     #0,
                     #0)
            xml_list.append(value)
    column_name = ['ImageID','Source','LabelName','Confidence','XMin','XMax','YMin','YMax']#,'IsOccluded','IsTruncated','IsGroupOf','IsDepiction','IsInside']
    df = pd.DataFrame(xml_list, columns=column_name)
    return df

def to_trainvaltest(pics, label_df):
    columns = label_df.columns
    labels = label_df.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(pics, labels,test_size=0.2, shuffle = True, random_state = 8)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=0.3, shuffle = True, random_state = 8)
    y_train = pd.DataFrame(y_train, columns=columns)
    y_test = pd.DataFrame(y_test, columns=columns)
    y_validation = pd.DataFrame(y_validation, columns=columns)
    return {'X_tr':X_train,'y_tr':y_train,'X_va':X_validation,'y_va':y_validation,'X_te':X_train,'y_te':y_test, 'columns':columns}

def df_to_csv(name, data):
    print(data)
    '''write the data!'''
    data.to_csv(name,sep=',', header=True)

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
                transformed = transform(image=img, bboxes=bbox)
                trans_img_bbox.append(transformed)
    return trans_img_bbox


def getTransformFunction(lossOfInfo=False):
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
    return transform    
