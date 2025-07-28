# CIFAR-10-CLASSIFICATION

#  CNN for Multiclass Image Classification using CIFAR-10

This repository contains a Convolutional Neural Network (CNN) implemented in **PyTorch** to classify images in the **CIFAR-10** dataset.

---

##  Dataset

- **CIFAR-10**: A labeled dataset of 60,000 32x32 color images across **10 classes**:
  - `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- Training: 50,000 images  
- Testing: 10,000 images  
- Automatically downloaded using `torchvision.datasets.CIFAR10`

---

##  Project Structure

- `cnn_multiclass_classification.ipynb`: CNN model code in PyTorch
- `CNN_CIFAR10_Report.pptx`: 10-slide presentation report
- `README.md`: Project overview

---

##  Model Architecture

Conv2D(3, 32) + ReLU + MaxPool  
Conv2D(32, 64) + ReLU + MaxPool  
Conv2D(64, 128) + ReLU + MaxPool  
Flatten  
Linear(2048, 256) + ReLU + Dropout  
Linear(256, 10) + Softmax


## Training Details
Optimizer: Adam

Loss Function: CrossEntropyLoss

Epochs: 10

Batch Size: 64

Device: GPU/CPU compatible

## Performance
Accuracy improved steadily across epochs

Final test accuracy: ~60–70% (depending on system)

Visualizations included for:

Loss vs Epochs

Accuracy vs Epochs

## Observations
CNN captures hierarchical features efficiently

Dropout prevents overfitting

Can be enhanced using data augmentation or deeper networks

