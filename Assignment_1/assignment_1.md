# Assignment 01 — Data Preprocessing and Visualization using TensorFlow/Keras

## Overview

This assignment introduces the basic workflow used when working with datasets for Deep Learning models. The implementation is performed using **TensorFlow and Keras in Google Colab**.

The assignment uses the **Iris dataset**, which contains measurements of iris flowers belonging to three different species: **Setosa, Versicolor, and Virginica**. The dataset contains four numerical features: sepal length, sepal width, petal length, and petal width.

The main purpose of this assignment is to understand how data is loaded, explored, preprocessed, normalized, divided into training and testing sets, and visualized before it is provided to a Deep Learning model.

---

## 1. TensorFlow and Keras Setup

The assignment begins by importing and configuring the required Deep Learning libraries:

* **TensorFlow** — used as the primary Deep Learning framework.
* **Keras** — used as the high-level API for building and working with Deep Learning models.
* **NumPy** — used for numerical operations.
* **Pandas** — used for data manipulation and analysis.
* **Matplotlib** — used for data visualization.
* **Scikit-learn** — used for dataset loading, train-test splitting, and feature scaling.

The notebook is executed in **Google Colab**, which provides a Python environment suitable for running TensorFlow and other machine learning libraries.

TensorFlow and Keras versions are also checked to verify that the environment has been configured correctly.

---

## 2. Loading the Dataset

The **Iris dataset** is loaded using the Scikit-learn dataset API.

The dataset contains **150 samples** with four numerical input features.

The features are:

* **Sepal Length**
* **Sepal Width**
* **Petal Length**
* **Petal Width**

The target variable represents the species of the iris flower.

The three classes are represented as:

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

Each sample therefore contains four input values and one corresponding target label.

---

## 3. Exploring the Dataset

Before performing preprocessing, the dimensions and contents of the dataset are inspected.

The notebook examines:

* Number of samples
* Number of features
* Column names
* Dataset information
* Statistical summary
* Target labels
* Missing values

Pandas is used to display and analyze the dataset in tabular form.

The `head()` function is used to view the first few records, while `describe()` provides statistical information such as mean, standard deviation, minimum, maximum, and quartile values.

The dataset is also checked for missing values to ensure that the input data is suitable for further processing.

---

## 4. Data Preprocessing

Data preprocessing is performed to prepare the dataset before it is given to a Deep Learning model.

The input features are separated from the target variable.

The four numerical features are used as the input:

```text
X → Input Features
y → Target Labels
```

The target column is separated from the feature columns so that the model can learn the relationship between the input features and the corresponding iris species.

The dataset is then prepared for train-test splitting and feature normalization.

---

## 5. Train-Test Split

The dataset is divided into training and testing sets using `train_test_split()` from Scikit-learn.

The dataset is divided using an **80:20 ratio**:

```text
Training Data → 80%
Testing Data  → 20%
```

This results in:

```text
Training samples → 120
Testing samples  → 30
```

The training data is used for learning patterns from the input features, while the testing data is kept separate for evaluating a future Deep Learning model on unseen examples.

The split uses a fixed `random_state` so that the same division can be reproduced when the notebook is executed again.

Stratification is also used to maintain a similar class distribution in both the training and testing sets.

---

## 6. Data Normalization

The original Iris features have different numerical ranges. For example, petal length and sepal width do not have exactly the same scale.

To make the features more suitable for Deep Learning, **StandardScaler** is used to standardize the input features.

Standardization transforms the features using the following formula:

```text
z = (x - μ) / σ
```

where:

* `x` represents the original feature value.
* `μ` represents the mean of the feature.
* `σ` represents the standard deviation of the feature.
* `z` represents the standardized value.

The standardized training features have approximately:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

The scaler is fitted only on the training data and then used to transform both the training and testing data.

This prevents information from the testing dataset from influencing the preprocessing process.

---

## 7. Data Visualization

The assignment uses **Matplotlib** to visualize the Iris dataset.

Different visualizations are created to understand the distribution and relationships between the features.

### Histogram

Histograms are used to observe the distribution of the numerical features.

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

This helps in understanding the spread and distribution of the values in the dataset.

### Scatter Plot

Scatter plots are used to examine relationships between different features.

For example:

```text
Sepal Length vs Sepal Width
Petal Length vs Petal Width
```

These visualizations help identify patterns and relationships between the input features.

The petal features provide a particularly useful visualization for observing differences between the iris classes.

---

## 8. TensorFlow Tensor Conversion

After preprocessing and normalization, the feature data can be converted into TensorFlow tensors.

The normalized NumPy arrays are converted using TensorFlow:

```text
X_train → TensorFlow Tensor
X_test  → TensorFlow Tensor
```

This demonstrates that the preprocessed data is now in a suitable format for use with TensorFlow/Keras models.

---

## 9. Concepts Covered

This assignment demonstrates the fundamental data preparation pipeline used in Deep Learning:

```text
Dataset
   ↓
Load Data
   ↓
Explore Dataset
   ↓
Check Missing Values
   ↓
Separate Features and Target
   ↓
Train-Test Split
   ↓
Normalize Features
   ↓
Visualize Data
   ↓
Convert to TensorFlow Tensors
   ↓
Ready for Deep Learning Model
```

These preprocessing steps form the foundation for subsequent assignments where the processed data can be used to train neural networks and other Deep Learning models.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

## Dataset

**Iris Dataset**

The Iris dataset consists of **150 samples** of iris flowers belonging to three different species.

The four input features are:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The target contains three classes:

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

## Google Colab

The assignment is implemented and executed using **Google Colab**.

(https://colab.research.google.com/drive/1IfgeNFza5E-KMc1YGEnguadFwBJqltPF?usp=sharing)

