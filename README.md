# Image Classification Model for War Zone Images

This repository contains two image classification models trained on a dataset of images labeled with five classes: "combat", "humanitarianaid", "militaryvehicles", "fire", and "destroyedbuilding". The models are:

1. A pre-trained ResNet18 model fine-tuned on our dataset.
2. A custom Convolutional Neural Network (CNN) model trained from scratch.

## Dataset

The dataset consists of images labeled into the following classes:
- `combat`
- `humanitarianaid`
- `militaryvehicles`
- `fire`
- `destroyedbuilding`

## Models

### 1. Pre-trained ResNet18

ResNet18 is a deep residual network with 18 layers. We used a model pre-trained on the ImageNet dataset and fine-tuned it on our specific dataset. Fine-tuning involves:
- Replacing the final layer with a new fully connected layer matching the number of classes in our dataset.
- Training the modified network on our dataset to adjust the weights of the final layer.

### 2. Custom CNN

Our custom CNN model was built from scratch to classify the images. The architecture includes several convolutional layers, pooling layers, and fully connected layers. This model was trained end-to-end on our dataset.

## Training and Evaluation

Both models were trained and evaluated on the same dataset. The training process involved:
- Data preprocessing: resizing images, normalization, and data augmentation (e.g., random cropping, flipping).
- Splitting the dataset into training and validation sets.
- Using cross-entropy loss as the loss function and Adam optimizer for training.

### Results

The performance of the models was evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Contributors

- [**Khush Patel**](https://github.com/khushpatel2121)
- [**Vishwa Pujara**](https://github.com/Vishwapujara)
- [**Nandani Thakkar**](https://github.com/nandani09)



