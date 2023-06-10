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

train, val, test = utils.to_trainvaltest(images, annotations,test_size=0.2, val_size=0.3)

txtlists = {}

factors=[10,3,3]
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
    print('Trainval written')

with open(os.getcwd()+"/data/Output/labels.txt", 'w') as fp:
    for item in labels:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Labels written')


#utils.dict_to_xml(label_dict,new_lab_to_path)
#here the pics should be loaded, ut lets see if it works with meandingless zeros
#pics = [0]*len(df)
#df = utils.matchPicsToLabels(df,pic_path)
#print(df)
'''
parts = utils.to_trainvaltest(pics, df)

utils.labels_creation(df)
utils.df_to_csv('label_validation.csv', parts['y_va'])
utils.df_to_csv('label_train.csv', parts['y_tr'])
utils.df_to_csv('label_test.csv', parts['y_te'])
'''