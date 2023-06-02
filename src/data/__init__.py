##Testfile for the image augmentation funtions
import os
import sys
from pathlib import Path
import image_augmentation as utils

pic_path = os.getcwd()+"/data/raw"
lab_path = os.getcwd()+"/data/labeling"

print(lab_path)

df = utils.xml_to_df(lab_path)
#here the pics should be loaded, ut lets see if it works with meandingless zeros
pics = [0]*len(df)
parts = utils.to_trainvaltest(pics, df)
'''for name, data in parts.items():
    print(name)
    print(data)'''
utils.df_to_csv('label_validation.csv', parts['y_va'])
utils.df_to_csv('label_train.csv', parts['y_tr'])
utils.df_to_csv('label_test.csv', parts['y_te'])
