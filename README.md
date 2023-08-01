Nvidia Jetson Nano - Assembly Assistant for Lego Blocks
==============================
This project is a proof of concept for an assembly assistant for Lego blocks. The goal of this project is to build and deploy a machine learning model that assists the user in assembling or disassembling a Lego object. This is assisted by a camera that detects the Lego blocks and highlights the next block to be placed. The user can then pick up the next block and place it on the Lego object. To guide the user, provide a better user experience and interaction, a web application is built. This web application shows the camera stream and the next block to be placed. The web application is built with Flask and Bootstrap. The machine learning model is built with the Jetson Inference library and deployed on a Nvidia Jetson Nano.

If you are first-time setting up the project we have a comprehensive guide to read through in our [Wiki](./wiki/home).

Running the project
------------
> **INFO** We provide a pre-configured Jetson Nano that let's you skip a lot of time in the installation process. If you have such a device, you only have to setup the client computer.

To get the project running some steps are required. First you need to have a configured Jetson Nano and client computer. They need to be connected and able to communicate with each other. The following wiki page provides you with all the necessary steps to get this done: [Installation Instructions](../../wikis/Installation). 

The next step is to start the web server with a camera and one of the models we provide. After the server has been started, the client computer can connect to the web application using any browser. In depth instructions for this are provided on the following wiki page: [Application Manual](../../wikis/Application-Manual). 

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

Contact
------------
For any questions or feedback you may contact us at:  
Marco Schneider - marco.schneider@student.kit.edu  - frontend implementation and configuration\
Leonard Ramadani - leonard.ramadani@student.kit.edu - Jetson Nano configuration and model training\
Anton - Image Augmentation\
Paul - Image Augmentation\
Patrick - patrick.roth@student.kit.edu - Image Augmentation
