# WhatPlane - A passenger aircraft recognition app

A computer vision app for Aircraft Recognition, built using PyTorch and deployed
using fastAPI.

Served using Docker on Azure App Service. Frontend is built using React and can
be found [here](https://github.com/what-plane/what-plane-frontend).

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── api
    │   ├── dependencies   <- Dependencies for the API
    │   ├── routers        <- Routers for the API (`/predict`, etc..)
    │   └── main.py        <- Main API function
    │  
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for experiments
    │
    ├── scripts            <- Scripts required for the project
    │
    ├── whatplane          <- Source code for the models and data.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data 
    │   │
    │   └── models         <- Scripts to train models
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
