import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from src.data.pipeline_utils import transformed


path = "../data/raw/nano"
json_file_path = "../data/labeling/nano_labeling.json"
img_nano = transformed(path, json_file_path, 1)

path = "../data/raw/smartphone"
json_file_path = "../data/labeling/smartphone_labeling.json"
img_sm = transformed(path, json_file_path, 1)



