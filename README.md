# Steam Game Popularity Predictor and Game Recommender

## Table of Contents
- [Introduction](#Introduction)
- [Packages Used](#Packages-Used)
- [Installation and Setup](#Installation-and-Setup)
- [Structure of the Project](#Structure-of-the-Project)
- [How to Run the Project](#How-to-Run-the-Project)

## Introduction
This project is designed to provide two primary services:

1. Train a Machine Learning model for game popularity prediction.
2. Recommend games similar to a given Steam game.

This project uses data from the Steam Store to perform these tasks. You can either train a Machine Learning model for game popularity prediction, or get recommendations for games similar to a game of your choice. 

## Packages Used
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computation.
- **sklearn**: For machine learning tasks, including model training and prediction.
- **seaborn**: For data visualization.
- **matplotlib**: For creating different kinds of visualizations in Python.
- **fuzzywuzzy**: For string matching, used to find the best match for a game name from the user's input.
- **python-Levenshtein**: An optional speedup for fuzzywuzzy.

## Installation and Setup
First, make sure you have Python 3.11 installed on your system. You can download it from the official Python website [here](https://www.python.org/downloads/).

Next, you will need to install several Python libraries. You can do this by running the following commands in your terminal:
```bash
pip install pandas
pip install numpy
pip install sklearn
pip install seaborn
pip install matplotlib
pip install fuzzywuzzy
pip install python-Levenshtein
```
After installing Python and the necessary libraries, clone the project repository to your local machine.

## Structure of the Project
This project is structured as follows:
- ``main.py``: This is the main script that you will run to use the project. It handles user interaction and calls the necessary functions based on user input.
- ``predict_popular.py``: This script contains the functionality for predicting a game's popularity. It includes its own data preprocessing, model training, and prediction functions.
- ``guess.py``: This script helps in finding the exact game name from the user's input.
- ``recommender.py``: This script contains the functionality for recommending similar games. It includes its own data preprocessing and recommendation functions.
- ``data``: This directory contains the data files used by the project.
- ``outputs``: This directory will be created when you run the program. It will contain the output files generated by the program.

## Cloning This Repository
This repository contains large files and uses [Git Large File Storage (LFS)](https://git-lfs.com). To clone this repository, you need to have Git LFS installed.

Follow these steps:
1. Install Git LFS. This only needs to be done once per machine:
For macOS, use Homebrew:
```bash
brew install git-lfs
```
For Windows, download the installer from the [Git LFS GitHub page](https://git-lfs.com).
For Linux, the process depends on the specific distribution.
2. Set up Git LFS for your user account. This also only needs to be done once:
```bash
git lfs install
```
3. Now you're ready to clone the repository. You can clone it as you normally would:
```bash
git clone git@github.com:alinikan/video-game-analysis.git
```
If you encounter any issues, Git LFS's GitHub page has [more detailed instructions and troubleshooting information](https://github.com/git-lfs/git-lfs/wiki/Tutorial).

## How to Run the Project
To run the project, navigate to the project's directory in your terminal and run the following command:
```bash
python3 main.py
```
The script will then guide you through the process. It will first ask you what you want to do. You can choose to either get a game recommendation based on your input, train a Machine Learning model for game popularity prediction, analyze the original data, or exit the program.

If you choose to get a game recommendation, you will be prompted to enter the name of a game. The script will then recommend games that are similar to the game you entered.

If you choose to train a Machine Learning model for game popularity prediction, the script will train and show you the scores of a model that can predict a game's popularity based on its characteristics.
Note that the GridsearchCV hyperparameter tuning can be turned on by changing the constant ``ENABLE_PARAM_TUNING`` to true in ``predict_popular.py``.

If you choose to analyze the original data, the script will show you some visualizations of the data (in the outputs folder), and will also show you some other information about the data.

## Contributors
- [Dingshuo Yang](https://github.com/HarukaYang)
- [Ali Nikan](https://github.com/alinikan)
- [Mohammad Parsaei](https://github.com/M-Parsaei)