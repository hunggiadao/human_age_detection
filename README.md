# Human Age Detection

## Introduction

This is a project with a goal of predicting the age of a person based on their facial features. The project employs a Convolutional Neural Network (CNN) model, inspired by Resnet50 CNN. Our methodology of implementation is based largely on Jakub Paplham and Vojtech Franc's *A Call to Reflect on Evaluation Practices for Age Estimation: Comparative Analysis of the State-of-the-Art and a Unified Benchmark* paper ([link](https://ieeexplore.ieee.org/document/10656298)).

The project utilizes the following design techniques:
- Sequential data loading using *DataLoader* from Python *Torch* library
- Data preprocessing: resizing, noise reduction, flipping, brightness adjustment
- Dataset splitting: training set (60%), validation set (20%), testing set (20%)
- Transfer learning using Resnet50 as a backbone model
- Checkpoints to save previously trained model

**Note:** this project requires Git Large File Storage (LFS) tracking for transfering files larger than 50MB.

## Setting up

Clone the repository:
```
$ git clone git@github.com:hunggiadao/human_age_detection.git
```

Install and use Git LFS:
Download and install [Git LFS](https://git-lfs.com).
Type the following command into terminal to enable LFS tracking in your current Git repository:
```
$ git lfs install
```

The provided `.gitattributes` should have all necessary LFS file types for this project. If you wish to include more file types for LFS tracking, you can add them directly to `.gitattributes` using the same syntax, or type this command into terminal:
```
$ git lfs track "*.<file-type>"
```
`*` means all files of this type.

## Description:

### Data collection:

We use ResNet50 pretrained model (pretrained on Image Net) to pretrain on 200,000 random images from 500,000 human face images of IMDB-wiki dataset for 2 epochs using an actual human face dataset (the resource is too limited to train on the whole dataset for longer epochs).

![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/data_collection.png)

### Data augmentation:

We observed that the UTKFace data distribution was not balanced. To compensate for a lack of images of senior populations, we applied data augmentation to make it more generalized, which increased the model performance.

![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/data%20augmentation.jpeg)

### Generalized data distribution:

Distribution of the world's population by age and sex, 2017. Source: *United Nations, Department of Economic and Social Affairs, Population Division (2017). World Population Prospects: The 2017 Revision. New York: United Nations*.

![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/data%20distribution.jpeg)

### Data preprocessing

1. Use MTCNN to detect the human face.
![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/preprocessing.png)
2. Resize the images to 256x256 resolution.
3. Randomly crop the images to reduce noise in training
4. Flip the images randomly
5. Adjust the brightness of the images with *ColorJitter*
6. Split our dataset (details in [Introduction]())




# Files
