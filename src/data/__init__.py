##Testfile for the image augmentation funtions
import os
import sys
from pathlib import Path
import image_augmentation as utils

pic_path = os.getcwd()+"/data/raw/"
lab_path = os.getcwd()+"/data/labeling/"

new_lab_path = os.getcwd()+"/data/Output/Annotations/"
new_pic_path = os.getcwd()+"/data/Output/JPEGImages/"

labels, annotations_dict = utils.xml_to_dict(lab_path)

images, annotations = utils.load_pics(pic_path,annotations_dict)

train, val, test = utils.to_trainvaltest(images, annotations,val_test_size=2/3, val_size=0.5)

txtlists = {}

factors=[10,1,1]
for item,factor in zip([train,val,test],factors):
    print('\n{}-set:\n'.format(item[0]))
    type, liste = utils.enlarge_dataset(factor, item,new_lab_path,new_pic_path)
    txtlists[type] = liste

    with open(os.getcwd()+"/data/Output/ImageSets/Main/"+ type + '.txt', 'w') as fp:
        for item in liste:
            # write each item on a new line
            fp.write("%s\n" % item)
        print(type+' written')

trainvallist = txtlists['train'] + txtlists['val']

with open(os.getcwd()+"/data/Output/ImageSets/Main/trainval.txt", 'w') as fp:
    for item in trainvallist:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('trainval written')

with open(os.getcwd()+"/data/Output/labels.txt", 'w') as fp:
    for item in labels:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('labels written')