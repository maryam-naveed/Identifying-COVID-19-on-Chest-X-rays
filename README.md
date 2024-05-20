# Identifying-COVID-19-on-Chest-X-rays

This project aims to classify chest X-ray images as either COVID-19 positive or normal using a Convolutional Neural Network (CNN). The dataset is sourced from Kaggle, and the implementation uses TensorFlow and Keras.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Features

- Downloads and extracts the COVID-19 chest X-ray image dataset from Kaggle.
- Splits the dataset into training and validation sets.
- Defines and trains a CNN model for binary classification.
- Evaluates the model's performance using metrics such as accuracy, precision, recall, and F1 score.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Kaggle API
- NumPy
- Scikit-learn

## Setup

1. **Kaggle API Configuration**

   - Download the Kaggle API token (`kaggle.json`) from your Kaggle account settings.
   - Upload `kaggle.json` to your working directory

.

2. **Install Required Packages**

   Install the necessary Python packages using pip:

   ```bash
   pip install tensorflow keras numpy scikit-learn kaggle
   ```

3. **Download and Extract Dataset**

   Configure your Kaggle API token and download the dataset:

   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d alifrahman/covid19-chest-xray-image-dataset
   ```

   Extract the downloaded dataset:

   ```bash
   unzip covid19-chest-xray-image-dataset.zip
   ```

4. **Prepare Data**

   Organize the dataset into training and validation directories:

   - Create directories for training and validation sets.
   - Split the images into these directories.

## Usage

1. **Create Data Generators**

   Utilize TensorFlow's `ImageDataGenerator` to preprocess the images for training and validation. This includes resizing the images and normalizing the pixel values.

2. **Define and Train the Model**

   Build a CNN model using Keras with the following layers:
   - Convolutional layers with ReLU activation and MaxPooling layers.
   - A Flatten layer followed by Dense layers for classification.

   Compile the model with the Adam optimizer and binary crossentropy loss, then train it using the training data.

3. **Evaluate the Model**

   After training, evaluate the model on the validation set to determine its accuracy, precision, recall, and F1 score. 

## Results

- **Validation Loss:** Display the final loss on the validation set.
- **Validation Accuracy:** Display the final accuracy on the validation set.
- **Precision:** Display the precision score.
- **Recall:** Display the recall score.
- **F1 Score:** Display the F1 score.
