Nvidia Jetson Nano - Assembly Assistant for Lego Blocks
==============================
This project is a proof of concept for an assembly assistant for Lego blocks. The goal of this project is to build and deploy a machine learning model that assists the user in assembling or disassembling a Lego object. This is assisted by a camera that detects the Lego blocks and highlights the next block to be placed. The user can then pick up the next block and place it on the Lego object. To guide the user, provide a better user experience and interaction, a web application is built. This web application shows the camera stream and the next block to be placed. The web application is built with Flask and Bootstrap. The machine learning model is built with the Jetson Inference library and deployed on a Nvidia Jetson Nano.


Installation
------------

To run the following commands you need to have Python 3 and pip installed. 
After pulling the repository, open the terminal and run the following commands.

#### Linux

```
pip install -e .
pip install -r requirements.txt
```

#### Windows

```
py -m pip install -e .
py -m pip install -r requirements.txt
```


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README
    │ 
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── frontend           <- Directory to store frontend code
    │   ├── data           <- Directory to store HTTPS certifications 
    |   ├── node_modules   <- Directory to store node modules for Bootstrap
    │   ├── static         <- Directory to store images, css and javascript files
    │   └── templates      <- Directory to store html files
    |
    ├── jetson-inference   <- The jetson-inference repository including the configured dockerfile
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts for data augmentation


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
