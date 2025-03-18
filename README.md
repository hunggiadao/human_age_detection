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

## Description

### Data collection:

We use ResNet50 pretrained model (pretrained on Image Net) to pretrain on 200,000 random images from 500,000 human face images of IMDB-wiki dataset for 2 epochs using an actual human face dataset (the resource is too limited to train on the whole dataset for longer epochs).

![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/data_collection.png)

### Data augmentation:

We observed that the UTKFace data distribution was not balanced. To compensate for a lack of images of senior populations, we applied data augmentation to make it more generalized, which increased the model performance.

![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/data%20augmentation.jpeg)

### Generalized data distribution:

Distribution of the world's population by age and sex, 2017. Source: *United Nations, Department of Economic and Social Affairs, Population Division (2017). World Population Prospects: The 2017 Revision. New York: United Nations*.

![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/data%20distribution.jpeg)

### Data preprocessing:

1. Use MTCNN to detect the human face
2. Align their faces straight
3. Crop the image to increase facial coverage
4. Resize the images to 256x256 resolution.
5. Randomly crop the images to reduce noise in training
6. Flip the images randomly
7. Adjust the brightness of the images with *ColorJitter*
8. Split our dataset (details in [Introduction](https://github.com/hunggiadao/human_age_detection/tree/main?tab=readme-ov-file#introduction))

### Model design:

1. Feature extraction:
-- Load the pretrained weights on IMDB (using `model_state_dict`) into our Resnet50 backbone layer.
2. Shared Fully Connected Layers:
-- 1024 → 512 neurons: Gradual reduction of feature space for efficient representation.
-- Regularization: Batch normalization and 40% dropout to prevent overfitting.
-- Non-linearity: ReLU activation enables learning complex relationships.
3. Age Prediction Head:
-- 512 → 128 → 1 neurons: Focuses on refining features for accurate regression output.
-- Final Output: Single continuous value for age prediction.
4. Freeze Layers until layer2.0.conv1:
-- Purpose: Gradually unfreezes layers to fine-tune specific parts of the model.
-- Mechanism: Freezes parameters until freeze_until layer is reached, then allows training.
-- Benefit: Preserves pre-trained knowledge while adapting to new tasks.

### Training:

1. Apply `AdamW`. Large weights can cause the model to memorize the training data, leading to poor generalization. Weight decay penalizes large weights, encouraging the model to learn simpler patterns that generalize better to unseen data
2. Apply `Scheduler` to dynamically adjust the learning rate during training based on validation performance.
3. Training Phase (20 epochs):
Reset gradients → Forward pass → Compute loss → Backpropagate → Update weights → Track training loss.
4. Validation Phase:
Forward pass on validation data → Compute loss → Track validation loss (no backpropagation or weight updates).
5. Print out the result:
Display progress using `tqdm` and print the epoch summary.

### Evaluation:

![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Presentation/evaluation%20results.png)

Overall metrics:
- MAE (Mean Absolute Error): Average of absolute differences between predicted and true values.
- MSE (Mean Squared Error): Average of squared differences between predicted and true values.
- RMSE (Root Mean Squared Error): Square root of MSE, interpretable on the same scale as the labels.
- Median Absolute Error (MedAE): Median of absolute differences, less sensitive to outliers.
- R² Score: How well predictions approximate the actual values (closer to 1 is better).

Prediction and Actual Value graph:
![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Report/EVALUATION/predictions_vs_actuals.png)

Residuals (errors) as a function of the True values:
![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Report/EVALUATION/residuals_plot.png)

Q-Q Plot:
![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Report/EVALUATION/qq_plot.png)

Plot metrics over batches:
![alt text](https://github.com/hunggiadao/human_age_detection/blob/main/Report/EVALUATION/metrics_over_batches.png)

## Reference

Jakub Paplhám and Franc, V. (2024).
*A Call to Reflect on Evaluation Practices for Age Estimation: Comparative Analysis of the State-of-the-Art and a Unified Benchmark.*
2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.1196–1205. doi:https://doi.org/10.1109/cvpr52733.2024.00120.

Sadek, I. (2017).
*Distribution of the world’s population by age and sex, 2017.*
Source: United Nations, Department of Economic and Social Affairs, Population Division (2017). World Population Prospects: The 2017 Revision. New York: United Nations.

‌Rajput, S. (2020).
*Face Detection using MTCNN.*\
Medium: https://medium.com/@saranshrajput/face-detection-using-mtcnn-f3948e5d1acb.
