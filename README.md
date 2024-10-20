# Bird Species Recognition with ResNet50
This project contains a machine learning model to classify bird species using a modified pre-trained ResNet50 model. The model is trained on a dataset of bird images and can be used to predict the bird species from new images. The project also includes a script for training the model and another script for recognizing birds from images or directories of images.

## Table of Contents

* Project Overview
* Requirements
* Setup and Installation
* Model Training
* Bird Recognition
* Dataset Structure
* Usage
* Model File
* Acknowledgments

## Project Overview

This project focuses on using a ResNet50 convolutional neural network (CNN) to classify bird species. The model is fine-tuned on a dataset containing images of various bird species. The trained model can recognize bird species from images provided by the user. There are two main components:

* `bird_model.py` : Used for training the model on bird species.
* `recognition.py` : Used for recognizing bird species from images or directories of images using the trained model.

## Requirements

Make sure you have the following dependencies installed:

* `Python 3.8+`
* `torch`
* `torchvision`
* `PIL` 
* `os`

You can install the dependencies using the following command:
* pip install torch torchvision Pillow

## Model Training

Script: `bird_model.py`

This script is used to train the ResNet50 model to recognize bird species. It uses the torchvision.models.resnet50 model pre-trained on the ImageNet dataset and fine-tunes it to classify a custom dataset of bird species.

* Data Augmentation: The training images undergo random transformations such as resizing, horizontal flips, and color jittering to make the model more robust.
* Loss Function: The model is trained using the CrossEntropyLoss.
* Optimizer: The `Adam` optimizer is used with a learning rate of 0.001.
* Early Stopping: If the loss does not significantly improve between epochs, the training stops early to prevent overfitting.
* After training, the model is saved as `bird_model.pth` and can be used for bird species recognition.

## Bird Recognition

Script: `recognition.py`
This script loads the pre-trained model (bird_model.pth) and predicts the bird species from new images. It can:

* Recognize bird species from a single image.
* Predict bird species for all images in a directory.

It uses the same data normalization steps as the training process and the pre-trained ResNet50 model for predictions.

## Dataset Structure

Your dataset should be structured in the following way:

```
birds_data/
    ├── train_data/
    │   ├── 001.Black_footed_Albatross/
    │   ├── 002.Laysan_Albatross/
    │   └── ... (other bird species folders)
    └── test_data/
        ├── 001.Black_footed_Albatross/
        ├── 002.Laysan_Albatross/
        └── ... (other bird species folders)

```


* `train_data` : Contains the training images, organized into subdirectories where each subdirectory represents a different bird species.
* `test_data` : Contains the testing images, organized in the same way as `train_data`.

### Usage

### Training the Model

To train the model, use the `bird_model.py` script. The script will train the model on the bird species dataset and save the trained model to `bird_model.pth`.
```
python bird_model.py
```
### Recognizing Birds
To recognize birds from an image or directory, use the recognition.py script.

* For a single image:
```
python recognition.py
```

You can replace the image path in the script with the path to the image you want to classify.

* For recognizing birds in a directory of images, update the `directory_path` in the `predict_birds_in_directory()` function.


## Model File

The trained model is saved as bird_model.pth. This file contains the model’s parameters (weights and biases) and can be loaded for further predictions or fine-tuning.

## Acknowledgments

`ResNet50` is a pre-trained model from PyTorch’s `torchvision` library, trained on the ImageNet dataset.
The bird species dataset was organized for this project to classify images of birds into 200 species.




  

