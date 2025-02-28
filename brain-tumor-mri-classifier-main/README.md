# Brain MRI Tumor Classification

Course project of **`Deep Learning`**  course - [MDSAA-DS](www.novaims.unl.pt/MDSAA-DS) - Spring 2023

## Details of the Project

### Abstract
The classification of brain tumors is performed by biopsy, which is not usually conducted before definitive brain surgery. 
The most common method for differential diagnostics of tumor type is magnetic resonance imaging (MRI). However, it is susceptible to human subjectivity, and a large amount of data is difficult for human observation.
Early brain–tumor detection mostly depends on the experience of the radiologist.
<p>The improvement of technology and machine learning can help radiologists in tumor diagnostics without invasive measures. A machine-learning algorithm that has achieved substantial results in image segmentation and classification is the convolutional neural network (CNN).

### Objective
The aim of this project is to present a new CNN architecture for brain tumor classification of three primary brain tumors gliomas, meningiomas, and pituitary.

## Methodology

### Dataset
Dataset used here is a combination of the following three datasets : 
- [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [SARTAJ](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no)

This dataset contains 7023 images of human brain MRI scans which are classified into 4 classes:
- glioma (Train: 1321, Test:300), 
- meningioma (Train: 1339, Test:306),
- pituitary (Train:1457, Test:300),
- no-tumor (Train:1595, Test:405).
>Dataset can be accessed on Kaggle [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

### Data Preparation
In Preparation step, dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) into the `downloads` directory and unzipped. A cropping technique was used to find the extreme top, bottom, left, and right points of the brain in each image. Then to read each original image, crop the brain part, and save the cropped image in the `Dataset` directory.


**Note: Kaggle API credentials were set up beforehand in order to download the dataset using the Kaggle API.**

### Data Preprocessing
In Preprocessing step, several operations performed to preprocess input images.
- First, it resizes the image to 256x256 pixels using cubic interpolation. 
- Then, it converts the image to float type and normalizes it by subtracting the mean and dividing by the standard deviation of pixel values. 
- The pixel values are then clipped to lie between 0 and 1. 
- Next, the function applies bilateral filtering to remove noise from the image.
- Finally, the image is converted back to uint8 type and returned. 
>This preprocessing step is intended to improve the quality of the input images and make them suitable for training a machine learning model.

- Extract images

In order to crop the part that contains only the brain of the image, We used a cropping technique to find the extreme top, bottom, left and right points of the brain. You can read more about it here [Finding extreme points in contours with OpenCV](https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/).

### Data Augmentation
In order to load our dataset and labeling each element :

- Read the images in gray.
- Preprocess the image
    - Resize the image to (256, 256) to feed it as an input to the neural network.
    - Convert the image to grayscale as contrast and texture info is most important in grayscale rather than RGB channels.
    - Apply normalization because we want pixel values to be scaled to the range 0-1.
    - Save processed images into new directory.

### Network Architecture
@TODO

## Getting Started

