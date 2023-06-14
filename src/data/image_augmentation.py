import cv2
import numpy as np
import os
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

    labels = set()
    anno_dict = {}
    
    anno_files = list(os.listdir(path))
    anno_files.remove('older labels')

    for xml_file in anno_files:
        '''one file corresponds to one picture'''
        tree = ET.parse(path+'/'+xml_file)
        root = tree.getroot()
        for member in root.findall('size'):
            height = float(member.find('height').text)
            width = float(member.find('width').text)

        bboxes = []
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = float(bbx.find('xmin').text)
            ymin = float(bbx.find('ymin').text)
            xmax = float(bbx.find('xmax').text)
            ymax = float(bbx.find('ymax').text)
            label = member.find('name').text

            bbox = [
            floor(xmin),
            floor(ymin),
            floor(xmax),
            floor(ymax),
            label
            ]

            bboxes.append(bbox)

        labels.add(label)

        filename = root.find('filename').text

        annotation = {'Filename':filename,
                     'Width':width,
                     'Height':height}

        annotation['bboxes'] = bboxes

        anno_dict[filename] = annotation

    #print(anno_dict)
    #print('--------------------')
    print(labels)
    return list(labels), anno_dict

############ Labels abspeichern #########################################

'''writes the labels in the right xml format'''
def dict_to_xml(annotation,to_path):

    object_data = annotation['bboxes']

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
    
    folder_element.text = 'Rad+Grau'
    filename_element.text = annotation['Filename']
    database_element.text = 'Unknown'
    annotation_element.text = 'Unknown'
    image_element.text = 'Unknown'
    width_element.text = str(annotation['Width'])
    height_element.text = str(annotation['Width'])
    depth_element.text = ''
    segmented_element.text = '0'
    
    for obj_data in object_data:
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
        
        name_element.text = obj_data[4]
        truncated_element.text = '0'
        occluded_element.text = '0'
        difficult_element.text = '0'
        xmin_element.text = str(obj_data[0])
        ymin_element.text = str(obj_data[1])
        xmax_element.text = str(obj_data[2])
        ymax_element.text = str(obj_data[3])
        attr_name_element.text = 'rotation'
        attr_value_element.text = '0.0'
    
    tree = ET.ElementTree(root_element)
    filepath = os.path.join(to_path, annotation['Filename'][:-4])
    tree.write(filepath+'.xml', encoding='utf-8')


    '''
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
    filename_element.text = str(annotation['Filename'])
    database_element.text = 'Unknown'
    annotation_element.text = 'Unknown'
    image_element.text = 'Unknown'
    width_element.text = str(int(annotation['Width']))
    height_element.text = str(int(annotation['Height']))
    depth_element.text = ''
    segmented_element.text = '0'
    name_element.text = annotation['bboxes'][0][-1]
    truncated_element.text = '0'
    occluded_element.text = '0'
    difficult_element.text = '0'
    xmin_element.text = str(int(annotation['bboxes'][0][0]))
    ymin_element.text = str(int(annotation['bboxes'][0][1]))
    xmax_element.text = str(int(annotation['bboxes'][0][2]))
    ymax_element.text = str(int(annotation['bboxes'][0][3])) 
    attr_name_element.text = 'rotation'
    attr_value_element.text = '0.0'

    tree = ET.ElementTree(root_element)
    filepath = os.path.join(to_path, annotation['Filename'][:-4])
    tree.write(filepath+'.xml', encoding='utf-8')
    '''


'''Splits the training set into train validation and test'''
def to_trainvaltest(pics, labels,test_size,val_size):

    X_train, X_test, y_train, y_test = train_test_split(pics, labels,test_size=test_size, shuffle = True, random_state = 8)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=val_size, shuffle = True, random_state = 8)
    
    return ('train',X_train,y_train),('val',X_validation,y_validation),('test',X_test,y_test)
 
def load_pics(path_img,annotations_dict):

    ordered_images = []
    ordered_annotations = []

    width=480
    height=640

    resizer = getTransformFunction(onlyResize=True)

    for filename in os.listdir(path_img):

        if filename not in annotations_dict:
            continue

        path_file = os.path.join(path_img, filename)

        image = cv2.imread(path_file)

        annotation = annotations_dict[filename]
        bboxes = annotation['bboxes']

        #plot_img_with_bbox(image, bboxes,message='Loaded picture '+filename)

        transformed = resizer(image=image, bboxes=bboxes,width=width,height=height)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        annotation['bboxes'] = transformed_bboxes

        annotation['Filename'] = filename
        annotation['Width'] = width
        annotation['Height'] = height

        ordered_images.append(transformed_image)
        ordered_annotations.append(annotation)

        print(filename+' is uploaded')

        #plot_img_with_bbox(transformed_image, transformed_bboxes,message='Loaded and resized picture '+filename)
    
    return ordered_images, ordered_annotations


def enlarge_dataset(factor, items, lab_path,image_path):
    
    type = items[0]
    images = items[1]
    annotations_dict = items[2]

    transformer = getTransformFunction()

    trainvaltest = []

    for image,annotation in zip(images,annotations_dict):
        bboxes = annotation['bboxes'].copy()
        original_filename = annotation['Filename']
        filename_split = original_filename.split(".")

        print('write {}'.format(filename_split[0]))
        
        new_image = image
        filenametxt = 'original_'+ filename_split[0]
        trainvaltest.append(filenametxt)
        filename = filenametxt + '.' + filename_split[1]
        
        save_augmentation(filename, new_image, annotation,lab_path,image_path)
        annotation['Filename'] = filename

        for i in range(1, factor):

            transformed = transformer(image=image, bboxes=bboxes)
            new_image = transformed["image"]
            annotation["bboxes"] = transformed["bboxes"]

            filenametxt = filename_split[0] + '_v' + str(i)
            trainvaltest.append(filenametxt)
            
            filename = filenametxt + '.' + filename_split[1]
            annotation['Filename'] = filename
            
            save_augmentation(filename, new_image, annotation,lab_path,image_path)
            
            
            
    print('Randomly chosen picture is shown...')
    plot_img_with_bbox(new_image, annotation["bboxes"], voc_pascal=True,message='Randomly chosen picture '+filename)

    return type, trainvaltest



def save_augmentation(filename,image, annotation,lab_path,pic_path):
        
    #plot_img_with_bbox(image, annotation["bboxes"], voc_pascal=True,message='This will be saved: '+filename)

    cv2.imwrite(
        pic_path + filename,
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )

    dict_to_xml(annotation, lab_path)


'''Get different Transform Functions that create loss in the data or not'''

def getTransformFunction(lossOfInfo=False,onlyResize=False,width=480,height=640):
    
    if onlyResize:
        transform = A.Compose(
        [
            A.Resize(height=height, width=width)
        ],
        bbox_params=A.BboxParams(format="pascal_voc"),
        )

    else:
        transform = A.Compose(
        [
            A.RandomCrop(height=580, width=450),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15,p=0.5),
            A.RandomFog(fog_coef_lower=0, fog_coef_upper=0.3,alpha_coef=0.05,p=0.5),
            A.Blur(blur_limit=3, always_apply=True, p=0.5),
            A.PixelDropout (dropout_prob=0.01, drop_value=0, p=0.8),
            A.PixelDropout (dropout_prob=0.1, drop_value=0, p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.8),
        ],
        bbox_params=A.BboxParams(format="pascal_voc"),
        )
    return transform 



'''Shows the image and bboxes on that image on the screen'''

def plot_img_with_bbox(new_image, bboxes,voc_pascal=True,message="Image with Bounding Box"):
    image = new_image.copy()

    for bbox in bboxes:
        if voc_pascal:
            x, y, xmax, ymax, label = bbox
        else:
            #assume coco
            x, y, width, height, label = bbox
            xmax = x+width
            ymax = y+height
        print(bbox)
        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(xmax), int(ymax)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image,
            label,
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    cv2.imshow(message, image)
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