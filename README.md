# Practicum Project 2025
**Georgia Institute of Technology**

**Student:** Tung Nguyen (tnguyen844@gatech.edu, tungtngyn@gmail.com)

**Sponsor:** Sentinel Devices

<br>

## Introduction
This repository hosts the code for the Sentinel Devices project at Georgia Tech. The goal of this project is to design and implement an anomaly detection system that is capable of:

1. Identifying anomalies in unlabeled sensor data (time-series).
2. Explain the anomaly detection flags in user-readable text.
3. Design a user interface.

The main focus of the project is on **interpretability**, with bonus points for designing a system that can run on minimal compute during inference.

<br>


## Dataset
The dataset used is the [MetroPT dataset](https://zenodo.org/records/6854240) and is publicly available.

<br>

## Repository Structure

This project requires access to the Open AI API. To be able to run some of the code, you must have a `.env` file with variable `OPENAI_API_KEY=sk-...`.

### Administrative Folders

* `01-docs`: Contains the MetroPT dataset's research paper + Sentinel Devices project syllabus.
* `output`: Stores output files from Jupyter notebooks.
* `raw-data`: Stores the raw CSV data from the MetroPT dataset. This repository is empty because the data does not fit into GitHub.

### Development Folders

* `00-dev-files`: Contains a series of Jupyter notebooks used during development and/or initial experimentation. 
 
The majority of the code in these Jupyter notebooks are not final and should be considered development code. 

Plots, models, and metrics in `03_modeling_f1.ipynb`, `04_metrics_f1.ipynb` are used in the final report and should be considered finalized code.

### Demo Files

The root folder contains the following files:

* `requirements.txt`: Contains libraries used in this project. This project was developed using Python's `venv` module and does not require `conda` installation.
* `00_setup_and_model_training.ipynb`: Contains processing code (e.g. to parse raw CSV and PDF data into databases), model training code (for anomaly detection model), and model inference code (for anomaly detection model).
* `01_inference.ipynb`: Contains code which leverages ChatGPT to convert anomaly detection data generated in `00_setup_and_model_training` into user-readable text.
* `02_unsupervised_model_tuning.ipynb`: Contains proof-of-concept code for parallel model training + model tuning via synthetic anomalies.  
  
### Streamlit App

A user interface was also created using `streamlit`. The app is fully contained in `04-app` and can be run using the following instructions:

1. Create and activate a virtual environment using the libraries in `requirements.txt`
2. Open the `04-app` folder in a Terminal window
3. Run `streamlit run app.py`

A browser window should open with the app running on `localhost`. The LLM has access the anomaly detection data generated in `00_setup_and_model_training` as well as plotting capabilities.

The application should look like this: 

![App Demo](04-app/imgs/App%20Demo.png)

When asked to plot, the LLM will generate the plot and save it into `/imgs`. The generated image will also be appended to the chat history:

![Plot Demo](04-app/imgs/Plot%20Demo.png)