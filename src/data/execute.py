import os
from pathlib import Path
import image_augmentation as utils

############### Data augmentation pipeline ##################################################

# Set paths
pic_path = os.getcwd()+"/data/raw/"
lab_path = os.getcwd()+"/data/labeling/"

new_lab_path = os.getcwd()+"/data/Output/Annotations/"
new_pic_path = os.getcwd()+"/data/Output/JPEGImages/"

# load labels
labels, annotations_dict = utils.xml_to_dict(lab_path)

# load images
images, annotations = utils.load_pics(pic_path,annotations_dict)

# split data
train, val, test = utils.to_trainvaltest(images, annotations,val_test_size=2/3, val_size=0.5)

'''factor: array [x,y,z] which states the increase of number of pictures in
training (x), validation (y) and test (z) '''

factors=[15,1,1]

# execute data augmentation
txtlists = {}
for dataset,factor in zip([train,val,test],factors):

    print('\n{}-set:\n'.format(dataset[0]))
    type, liste = utils.enlarge_dataset(factor, dataset,new_lab_path,new_pic_path)
    txtlists[type] = liste

    # write the txt list which state which picture belongs to the train, validation and test set
    with open(os.getcwd()+"/data/Output/ImageSets/Main/"+ type + '.txt', 'w') as fp:
        for item in liste:
            fp.write("%s\n" % item)
        print(type+' written')

# write additional txt sheets that states wchich are the train and validation images

trainvallist = txtlists['train'] + txtlists['val']

with open(os.getcwd()+"/data/Output/ImageSets/Main/trainval.txt", 'w') as fp:
    for item in trainvallist:
        fp.write("%s\n" % item)
    print('trainval written')

# write txt sheet that lists all detected label names in the annotations

with open(os.getcwd()+"/data/Output/labels.txt", 'w') as fp:
    for item in labels:
        fp.write("%s\n" % item)
    print('labels written')