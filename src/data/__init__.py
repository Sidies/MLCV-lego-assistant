##Testfile for the image augmentation funtions
import os
import sys
from pathlib import Path
import image_augmentation as utils

pic_path = os.getcwd()+"/data/raw"
lab_path = os.getcwd()+"/data/labeling"

new_lab_path = os.getcwd()+"/data/labeling/augmented/"
new_pic_path = os.getcwd()+"/data/processed/"

labels_dict = utils.xml_to_dict(lab_path)

images, labels = utils.load_pics(pic_path,labels_dict)

train, val, test = utils.to_trainvaltest(images, labels,test_size=0.2, val_size=0.3)

factors=[5,5,5]
for item,factor in zip([train,val,test],factors):
    print('\n{}-set:\n'.format(item[0]))
    utils.enlarge_dataset(factor, item,new_lab_path,new_pic_path)

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