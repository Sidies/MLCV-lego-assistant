AISS_jetson_nano
==============================

to-do

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
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
