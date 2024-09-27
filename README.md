SHM-seismic-events-detection-with-bayesian-fusion
=====

![PyTorch](https://img.shields.io/badge/PyTorch-red.svg)

Project Overview
--
This repository hosts the implementation of automated seismic event detection under faulty data interference for structural health monitoring. The system primarily utilizes deep learning models to classify data types within each sensor channel. Subsequently, the multi-channel results are then integrated by a Bayesian fusion algorithm, achieving
automated and consistent seismic event detection.

Data and Dependencies
--
**Data Confidentiality:** Due to strict confidentiality agreements, this repository does not include original datasets from the bridge's Structural Health Monitoring (SHM) system.

**Included Data:** This repository contains detection results from deep learning models and manual labels for four seismic events. Each event is represented by a 70x248 matrix, where 70 denotes the number of sensor channels, and 248 represents the time-frequency (T-F) plots generated per channel per hour. Each matrix element corresponds to the data type at that specific time. Those data are located in the `bayesian_fusion` directory and are readily usable as inputs for Bayesian fusion algorithms `predict_using_previous_likeli.m`.

**Dependencies:** All package dependencies required for running the program are listed in the `requirements.txt` file.

