
# Face Mask Detection Dashboard

This project is a computer vision classification model designed to detect whether a person is wearing a face mask correctly, incorrectly, or not at all. It features a deep learning pipeline built with TensorFlow/Keras and includes a web-based interactive dashboard for real-time inference.

## Overview

The model uses transfer learning to classify images into three distinct categories:

* **`with_mask`**: The mask is worn properly covering the nose and mouth.


* **`without_mask`**: No mask is present.


* **`incorrect_mask`**: The mask is worn improperly (e.g., on the chin or not covering the nose).



## Technical Details

* **Model Architecture**: The core of the model utilizes **MobileNetV2** pre-trained on ImageNet for efficient feature extraction.


* **Custom Top Layers**: Added custom layers including `AveragePooling2D`, `Flatten`, and fully connected `Dense` layers (256 and 128 units) with `Dropout` for regularization to prevent overfitting.


* **Data Augmentation**: To improve model robustness, the training pipeline uses `ImageDataGenerator` to apply random rotations, zooms, width/height shifts, shears, and horizontal flips.


* **Training Strategy**:
* The dataset is split into 70% training, 15% validation, and 15% testing sets.


* **Phase 1**: Trained the custom head while keeping the MobileNetV2 base model frozen using the Adam optimizer.


* **Phase 2**: Fine-tuned the last 30 layers of the base model with a reduced learning rate to boost accuracy.




* **Callbacks**: Implemented `ModelCheckpoint` to save the best weights, `EarlyStopping` to halt training if validation loss plateaus, and `ReduceLROnPlateau` to dynamically adjust the learning rate.



## Evaluation Metrics

The notebook includes a comprehensive evaluation suite:

* Generates a **Classification Report** detailing precision, recall, and f1-score for each class.


* Plots a **Confusion Matrix** to visualize the true vs. predicted performance across the three categories.


* Visualizes the **Training/Validation Accuracy and Loss curves** across both the initial training and fine-tuning phases.



## Web Interface Deployment

This project utilizes **Gradio** to spin up an interactive web dashboard. Users can upload an image directly to the interface and instantly receive predictions alongside a confidence score for each class.

## How to Use

1. Open the Google Colab Notebook (`.ipynb`).
2. Ensure your dataset zip file is mounted correctly in Google Drive.


3. Run all cells to extract the data, train the model, and evaluate the results.


4. The final cell will launch the Gradio web interface. Click the public share link provided in the output to access the dashboard from any browser.


5. The trained model weights are saved and can be downloaded as `mask_detector.h5`.
