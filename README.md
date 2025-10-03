# Prodigy InfoTech ML Internship - Task 03
# By: Zain Qamar

## Cat & Dog Image Classifier using SVM

### Project Overview

This project implements a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. The project is split into two main parts: a script for training and saving the model, and a second script for loading the saved model to make predictions on new, unseen images. This repository contains the solution for **Task 03** of the Prodigy InfoTec Machine Learning Internship.

---

### Workflow

The project operates in two distinct phases:

#### Part 1: Model Training (`SVM_image_classifier_20000_samples.ipynb`)

1.  **Data Loading & Preprocessing:**
    *   A random sample of 20,000 images is loaded from the `cats` and `dogs` training directories.
    *   Each image is converted to grayscale and resized to `64x64` pixels for computational efficiency.
    *   The 2D image is flattened into a 1D vector and its pixel values are normalized to a range of 0-1.

2.  **Model Training:**
    *   The preprocessed data is split into training (80%) and validation (20%) sets.
    *   A **Support Vector Classifier (SVC)** with an `rbf` kernel is trained on the training data.

3.  **Evaluation & Saving:**
    *   The model's performance is evaluated on the validation set to check its accuracy, precision, and recall.
    *   The trained classifier is saved to a file (`svm_cat_dog_classifier_20k.joblib`) using `joblib` for later use.

#### Part 2: Prediction (`trained_model_loading_and_testing.ipynb`)

1.  **Load Model:** The pre-trained SVM model is loaded from the `.joblib` file.
2.  **Process Test Images:** Twenty random images are selected from the test directory. They undergo the *exact same* preprocessing steps (grayscale, resize, flatten, normalize) as the training images.
3.  **Predict & Visualize:** The model predicts whether each image is a 'Cat' or a 'Dog', and the results are displayed in a grid using Matplotlib.

---

### Performance & Results

#### Validation Performance

The model achieved a validation accuracy of approximately **64.65%**.


#### Prediction on Test Images

The following is a sample of the model's predictions on 20 unseen test images.


---

### How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/prime-programmer-ar/PRODIGY_ML_03
    cd PRODIGY_ML_03
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the dataset:**
    *   Create a `dataset` directory.
    *   Inside `dataset`, create a `train` folder. Inside `train`, create `cats` and `dogs` subfolders and place the respective training images there.
    *   Inside `dataset`, create a `test` folder and place unlabeled test images inside.

4.  **Execute the scripts in order:**
    *   **First, run the training notebook:**
      `SVM_image_classifier_20000_samples.ipynb`
      *(This will take a significant amount of time and will create the `svm_cat_dog_classifier_20k.joblib` file.)*
    *   **Then, run the prediction notebook:**
      `trained_model_loading_and_testing.ipynb`
      *(This will load the saved model and show the prediction grid.)*

---

### Libraries Used
- Scikit-learn (for SVM and metrics)
- OpenCV (cv2) (for image processing)
- NumPy (for array manipulation)
- Matplotlib (for visualization)
- joblib (for saving/loading the model)
- tqdm (for progress bars)