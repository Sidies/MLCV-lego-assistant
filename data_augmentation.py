import cv2
import os
import pandas as pd
import numpy as np
import albumentations as A
import json

def draw_rect(img, bboxes, color=(255, 0, 0)):
    img = img.copy()
    for bbox in bboxes:
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        img = cv2.rectangle(img, pt1, pt2, color, 2)
    return img

with open('via_project_20May2023_15h27m.json') as json_file:
    data = json.load(json_file)
 
    img_json = data["_via_img_metadata"]
 
    # Print the data of dictionary
    # print(json.dumps(data,sort_keys=True, indent=6))

img_meta = {}
for key, item in img_json.items():
    pic_name = str(key).split('.jpg')[0]
    boxes_json = item['regions']
    bboxes = []
    for bbox in boxes_json:
        shape = bbox['shape_attributes']
        tag = bbox['region_attributes']['tag']
        bboxes.append([shape['x'],shape['y'],shape['width'],shape['height'],tag])
    img_meta[pic_name] = bboxes
    
print(img_meta)

transform = A.Compose([
    #A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.8),
    A.RandomBrightnessContrast(p=0.4),
    #A.RandomFog(p=0.6),
    A.Rotate(p=1, limit=10)
], bbox_params=A.BboxParams(format='coco'))

for pic_name in img_meta.keys():
    file = os.getcwd()+'/'+pic_name+'.jpg'
    image = cv2.imread(file)
    bboxes = img_meta[pic_name]

    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    
    # Change pixel position from float to int
    int_trans_bboxes = []
    for bbox in transformed_bboxes:
        int_trans_bboxes.append([int(numb) for numb in bbox[:4]]+[bbox[4]])

    print(int_trans_bboxes)

    image = draw_rect(image, bboxes)
    transformed_image = draw_rect(transformed_image,int_trans_bboxes)
    cv2.imshow('original image', image)
    cv2.imshow('transformed image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
