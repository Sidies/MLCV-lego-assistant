import cv2
import numpy as np
import os
#import xmltodict
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import pandas as pd
import albumentations as A
from math import floor

########## Label laden #############################################

'''Gets all the labels of one directory and
returns them as dictionary with Filename as key'''
def xml_to_dict(path):

    label_dict = {}
    
    label_files = list(os.listdir(path))
    label_files.remove('augmented')
    for xml_file in label_files:
        '''one file corresponds to one picture'''
        tree = ET.parse(path+'/'+xml_file)
        root = tree.getroot()
        for member in root.findall('size'):
            height = float(member.find('height').text)
            width = float(member.find('width').text)
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = float(bbx.find('xmin').text)
            ymin = float(bbx.find('ymin').text)
            xmax = float(bbx.find('xmax').text)
            ymax = float(bbx.find('ymax').text)
            label = member.find('name').text

        filename = root.find('filename').text

        xml_data = {'Filename':filename,
                     'LabelName':label,
                     'Width':width,
                     'Height':height,
                     'XMin':xmin,
                     'XMax':xmax,
                     'YMin':ymin,
                     'YMax':ymax}
        bbox_data = [
            floor(xmin),
            floor(ymin),
            floor(xmax),
            floor(ymax),
            label
        ]
        label_dict[filename] = {'xml_data':xml_data, 'bbox_data': bbox_data}

    return label_dict

############ Labels abspeichern #########################################

'''writes the labels in the right xml format'''
def dict_to_xml(bboxOfPic,to_path):

    root_element = ET.Element('annotation')
    
    folder_element = ET.SubElement(root_element, 'folder')
    filename_element = ET.SubElement(root_element, 'filename')
    source_element = ET.SubElement(root_element, 'source')
    database_element = ET.SubElement(source_element, 'database')
    annotation_element = ET.SubElement(source_element, 'annotation')
    image_element = ET.SubElement(source_element, 'image')
    size_element = ET.SubElement(root_element, 'size')
    width_element = ET.SubElement(size_element, 'width')
    height_element = ET.SubElement(size_element, 'height')
    depth_element = ET.SubElement(size_element, 'depth')
    segmented_element = ET.SubElement(root_element, 'segmented')
    object_element = ET.SubElement(root_element, 'object')
    name_element = ET.SubElement(object_element, 'name')
    truncated_element = ET.SubElement(object_element, 'truncated')
    occluded_element = ET.SubElement(object_element, 'occluded')
    difficult_element = ET.SubElement(object_element, 'difficult')
    bndbox_element = ET.SubElement(object_element, 'bndbox')
    xmin_element = ET.SubElement(bndbox_element, 'xmin')
    ymin_element = ET.SubElement(bndbox_element, 'ymin')
    xmax_element = ET.SubElement(bndbox_element, 'xmax')
    ymax_element = ET.SubElement(bndbox_element, 'ymax')
    attributes_element = ET.SubElement(object_element, 'attributes')
    attribute_element = ET.SubElement(attributes_element, 'attribute')
    attr_name_element = ET.SubElement(attribute_element, 'name')
    attr_value_element = ET.SubElement(attribute_element, 'value')

    folder_element.text = ''
    filename_element.text = str(bboxOfPic['Filename'])
    database_element.text = 'Unknown'
    annotation_element.text = 'Unknown'
    image_element.text = 'Unknown'
    width_element.text = str(bboxOfPic['Width'])
    height_element.text = str(bboxOfPic['Height'])
    depth_element.text = ''
    segmented_element.text = '0'
    name_element.text = bboxOfPic['LabelName']
    truncated_element.text = '0'
    occluded_element.text = '0'
    difficult_element.text = '0'
    xmin_element.text = str(bboxOfPic['XMin'])
    ymin_element.text = str(bboxOfPic['YMin'])
    xmax_element.text = str(bboxOfPic['XMax'])
    ymax_element.text = str(bboxOfPic['YMax']) 
    attr_name_element.text = 'rotation'
    attr_value_element.text = '0.0'

    tree = ET.ElementTree(root_element)
    filepath = os.path.join(to_path, bboxOfPic['Filename'][:-4])
    tree.write(filepath+'.xml', encoding='utf-8')



'''Splits the training set into train validation and test'''
def to_trainvaltest(pics, labels,test_size,val_size):
    labels = labels
    X_train, X_test, y_train, y_test = train_test_split(pics, labels,test_size=test_size, shuffle = True, random_state = 8)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=val_size, shuffle = True, random_state = 8)
    #y_train = pd.DataFrame(y_train, columns=columns)
    #y_test = pd.DataFrame(y_test, columns=columns)
    #y_validation = pd.DataFrame(y_validation, columns=columns)
    return ('train',X_train,y_train),('validation',X_validation,y_validation),('test',X_test,y_test)
 
def load_pics(path_img,labels_dict):
    images = []
    labels = []
    for filename in os.listdir(path_img):
        path_file = os.path.join(path_img, filename)
        images.append(cv2.imread(path_file))
        labels.append(labels_dict[filename])
        print(filename+' is uploaded')
    
    return images, labels


def enlarge_dataset(factor, items, lab_path,pic_path):
    
    folder = items[0]
    pics = items[1]
    labels_dict = items[2]

    pic_path = pic_path + folder + '/'
    lab_path = lab_path + folder+'/'

    transformer = getTransformFunction()

    for image,labelData in zip(pics,labels_dict):
        bbox = labelData['bbox_data']
        bboxes = [bbox]
        label = labelData['xml_data']
        original_filename = label['Filename']
        filename_split = original_filename.split(".")

        print('write {}'.format(filename_split[0]))
        
        save_augmentation(filename_split, image, label,bboxes,lab_path,pic_path,i=0)
        
        for i in range(1, factor):
            
            transformed = transformer(image=image, bboxes=bboxes)
            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            
            save_augmentation(filename_split, transformed_image, label,transformed_bboxes,lab_path,pic_path,i=i)
            
            #plot_img_with_bbox(transformed_image, transformed_bboxes, voc_pascal=True)


def save_augmentation(filename_split,image, label,bboxes,lab_path,pic_path,i):
    #print('write version {}'.format(i))
    filename = filename_split[0] + '_v' + str(i) + '.' + filename_split[1]
    cv2.imwrite(
        pic_path + filename,
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )

    xMin = bboxes[0][0]
    yMin = bboxes[0][1]
    xMax = bboxes[0][2]
    yMax = bboxes[0][3]

    label['Filename'] = filename
    label['LabelName'] = bboxes[0][4]

    label['XMin'] = xMin
    label['YMin'] = yMin
    label['XMax'] = xMax
    label['YMax'] = yMax

    label['Width'] = xMax - xMin
    label['Height'] = yMax - yMin
    dict_to_xml(label, lab_path)

'''Get different Transform Functions that create loss in the data or not'''

def getTransformFunction(lossOfInfo=False):
    transform = A.Compose(
    [
        A.Resize(height=480, width=640),
        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90,p=0.9),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc"),
    )
    return transform 



'''Shows the image and bboxes on that image on the screen'''

def plot_img_with_bbox(transformed_image, transformed_bboxes,voc_pascal=False):
    if voc_pascal:
        x, y, xmax, ymax, label = transformed_bboxes[0]
    else:
        #assume coco
        x, y, width, height, label = transformed_bboxes[0]
        xmax = x+width
        ymax = y+height
    cv2.rectangle(
        transformed_image,
        (int(x), int(y)),
        (int(xmax), int(ymax)),
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

#######################################################

################ Back Up ##############################
#######################################################

######## Functions that are not used any more #########


'''Wo noch csv Label Dateien verwendet wurden:'''

def df_to_csv(name, data):
    #print(data)
    '''write the data!'''
    data.to_csv(name,sep=',', header=True)

'''Erstellt eine csv die alle Label Arten auflistet: blau, grau, rad, ...'''

def labels_creation(df):
    print(df)
    labeltypes = df['LabelName'].unique()
    # because we have no extra identifier, the label name is the label identifie
    
    labeltypes = pd.DataFrame(labeltypes, columns=['LabelName'])
    labeltypes['labels_duplicate'] = labeltypes.loc[:, 'LabelName']
    print(labeltypes)
    df.to_csv('class-descriptions-boxable.csv',sep=',', header=False)


'''Wo die Label Daten als json gespeichert waren'''

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
                    # Get Key of "region_attributes "
                    label = json_data[key]["regions"][0]["region_attributes"]["label"]

                    # coco [x_min, y_min, width, height]
                    bboxes = [[x, y, width, height, label]]
                    return bboxes


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



'''Approach that matches pics to labels and saves it structured,
but that costs to much RAM'''
def matchPicsToLabels(df,path_img):
    os.listdir(path_img)
    images=[]
    for k, filename in zip(df.index,df['Filename']):
        path_file = os.path.join(path_img, filename)
        images.append(cv2.imread(path_file))
        print('Pics loaded: '+str(k))
    return df