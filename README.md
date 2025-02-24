# Scene Classification with ResNet-34

## Overview
This project focuses on training a deep learning model using **ResNet-34** for **scene classification**. The dataset consists of images categorized into **40 scene classes**. The model is fine-tuned using a pre-trained ResNet-34 to improve classification accuracy. 

Key aspects of the project include:
- **Training and validation performance tracking** using TensorBoard
- **Evaluation metrics** including **Top-1 and Top-5 accuracy**
- **Confusion matrix visualization** to analyze misclassifications

## Dataset
- **Dataset Source**: Places2 Dataset (Simplified version)
- **Number of Classes**: 40
- **Data Augmentation Techniques**:
  - Random Rotation
  - Color Jitter
  - Normalization
- **Data Splitting**: 
  - 80% Training
  - 20% Validation

## Model Architecture
- **Backbone**: ResNet-34 (Pretrained on ImageNet)
- **Custom Layers**:
  - Fully connected layer with **256 hidden units**
  - **ReLU activation**
  - Final classification layer with **40 output classes**

## Training Details
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW
- **Learning Rate Strategy**:
  - **0.01** for the final classification layer
  - **0.001** for earlier layers
  - Step decay every **5 epochs** (gamma=0.1)
- **Batch Size**: 16
- **Epochs**: 10

## Evaluation Metrics
- **Top-1 Accuracy**: Measures the percentage of correct classifications
- **Top-5 Accuracy**: Measures how often the correct class is in the top 5 predictions
- **Confusion Matrix Visualization**: Helps analyze misclassifications and model performance

## Results
- The model achieves strong classification performance.
- **Top-1 and Top-5 accuracy** are used as evaluation metrics.
- **Confusion matrix plotted** to analyze misclassified images.
- **Randomly selected validation images** are visualized along with their predictions.

## Usage in Google Colab
To run this project in **Google Colab**, follow these steps:

1. Open Google Colab and upload the `.ipynb` file.
2. Download the datset zip file
3. Run the notebook cells sequentially to:
   - Load and preprocess the dataset
   - Train the model
   - Evaluate performance
   - Visualize results


## Contributions
Contributions are encouraged! Feel free to fork this repository, report issues, and submit pull requests. This is a great starting point for anyone looking to run machine learning code.
