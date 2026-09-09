# Assignment 02 — Multilayer Perceptron (MLP) for Classification

## Overview

This assignment introduces the implementation of a **Multilayer Perceptron (MLP)** for classification using **TensorFlow and Keras**.

The assignment uses the **Iris dataset**, which contains measurements of iris flowers belonging to three different species: **Setosa, Versicolor, and Virginica**.

The main purpose of this assignment is to understand how a neural network can be designed and trained for a classification problem. The assignment covers data preprocessing, feature normalization, train-test splitting, MLP architecture design, model training, prediction, and performance evaluation using **accuracy and a confusion matrix**.

---

## 1. TensorFlow and Keras Setup

The assignment begins by importing the required libraries for building and training the neural network.

The following libraries are used:

* **TensorFlow** — used as the primary Deep Learning framework.
* **Keras** — used to create and train the MLP model.
* **NumPy** — used for numerical operations.
* **Pandas** — used for data handling and analysis.
* **Matplotlib** — used for visualization.
* **Scikit-learn** — used for dataset loading, train-test splitting, preprocessing, and evaluation.

The implementation is performed in **Google Colab**, which provides a Python environment for running TensorFlow and Keras models.

---

## 2. Loading the Dataset

The **Iris dataset** is used for the classification task.

The dataset contains **150 samples** of iris flowers with four numerical input features.

The features are:

* **Sepal Length**
* **Sepal Width**
* **Petal Length**
* **Petal Width**

The target variable represents the species of the flower.

The three classes are:

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

The goal of the MLP is to learn the relationship between the four input features and predict the correct iris species.

---

## 3. Exploring the Dataset

Before training the model, the dataset is inspected to understand its structure.

The notebook checks:

* Number of samples
* Number of input features
* Shape of the dataset
* Feature values
* Target labels
* Statistical information

The dataset is also checked for missing values.

This exploration helps ensure that the dataset is properly understood and prepared before it is passed to the neural network.

---

## 4. Data Preprocessing

The input features and target labels are separated before training the model.

The four numerical features are used as input:

```text
X → Input Features
y → Target Labels
```

The target labels represent the three Iris classes.

The dataset is then divided into training and testing sets.

---

## 5. Train-Test Split

The Iris dataset is divided into training and testing data using an **80:20 split**.

```text
Training Data → 80%
Testing Data  → 20%
```

This results in:

```text
Training samples → 120
Testing samples  → 30
```

The training data is used to train the MLP, while the testing data is kept separate to evaluate the model on unseen samples.

A fixed `random_state` is used to make the split reproducible.

Stratification is also used so that the class distribution remains balanced between the training and testing datasets.

---

## 6. Feature Normalization

Feature normalization is performed before training the MLP.

The Iris features have different numerical ranges, so **StandardScaler** is used to standardize the input features.

The standardization formula is:

```text
z = (x - μ) / σ
```

where:

* `x` is the original feature value.
* `μ` is the mean of the training feature.
* `σ` is the standard deviation of the training feature.
* `z` is the standardized feature value.

The scaler is fitted only on the training data:

```text
Training Data
      ↓
Fit StandardScaler
      ↓
Transform Training Data
      ↓
Transform Testing Data
```

This prevents information from the testing dataset from influencing the training process.

---

## 7. MLP Architecture

A **Multilayer Perceptron** is a feed-forward artificial neural network consisting of an input layer, one or more hidden layers, and an output layer.

The MLP used in this assignment consists of:

```text
Input Layer
    ↓
Dense Hidden Layer
    ↓
Dense Hidden Layer
    ↓
Output Layer
```

The input layer receives the four Iris features.

The hidden layers use the **ReLU (Rectified Linear Unit)** activation function to introduce non-linearity into the network.

The output layer contains **3 neurons**, corresponding to the three Iris classes.

The output layer uses the **Softmax** activation function to produce class probabilities.

The class with the highest probability is selected as the predicted class.

---

## 8. Building the MLP Model

The MLP is implemented using the Keras `Sequential` API.

A typical architecture used in this assignment is:

```text
Input Layer       → 4 features
Hidden Layer 1    → 16 neurons, ReLU
Hidden Layer 2    → 8 neurons, ReLU
Output Layer      → 3 neurons, Softmax
```

The model can be created using:

```python
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])
```

---

## 9. Model Compilation

Before training, the model is compiled using an optimizer, loss function, and evaluation metric.

The following configuration is used:

```text
Optimizer → Adam
Loss      → Sparse Categorical Crossentropy
Metric    → Accuracy
```

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Loss Function

**Sparse Categorical Crossentropy** is used because the target labels are represented as integer class values:

```text
0, 1, 2
```

### Optimizer

The **Adam optimizer** is used to update the neural network weights during training.

### Accuracy

Accuracy measures the percentage of correctly classified samples.

---

## 10. Model Training

The MLP is trained using the training dataset.

```python
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
```

During training, the model learns patterns from the input features and adjusts its weights to reduce the classification error.

The training process can be visualized using training and validation accuracy.

---

## 11. Training Visualization

The training history is used to visualize how the model's accuracy changes during training.

```python
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.show()
```

The graph helps observe whether the model is learning effectively during training.

---

## 12. Model Evaluation

After training, the model is evaluated using the testing dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("Test Accuracy:", test_accuracy)
```

The **test accuracy** represents the percentage of test samples correctly classified by the trained MLP.

For example:

```text
Test Accuracy: 0.96
```

This represents approximately:

```text
96% classification accuracy
```

The exact accuracy may vary depending on the train-test split, model architecture, and training configuration.

---

## 13. Making Predictions

The trained MLP is used to predict the classes of the test samples.

```python
y_pred_prob = model.predict(X_test_scaled)

y_pred = np.argmax(y_pred_prob, axis=1)
```

The model initially produces probabilities for each of the three classes.

For example:

```text
Setosa       → 0.01
Versicolor   → 0.97
Virginica    → 0.02
```

The class with the highest probability is selected as the final prediction.

```text
Predicted Class → Versicolor
```

---

## 14. Confusion Matrix

A **confusion matrix** is used to evaluate the classification performance of the MLP.

It compares the actual class labels with the predicted class labels.

The confusion matrix contains:

```text
                 Predicted
              0     1     2
Actual  0     TP    ...   ...
        1     ...   TP    ...
        2     ...   ...   TP
```

For a multi-class classification problem, each row represents the actual class and each column represents the predicted class.

A correct prediction appears along the **main diagonal** of the confusion matrix.

---

## 15. Generating the Confusion Matrix

The confusion matrix is generated using Scikit-learn.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

The matrix can also be visualized using Matplotlib.

```python
plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1, 2],
    ["Setosa", "Versicolor", "Virginica"]
)

plt.yticks(
    [0, 1, 2],
    ["Setosa", "Versicolor", "Virginica"]
)

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()
```

---

## 16. Interpreting the Confusion Matrix

The confusion matrix helps identify which classes are correctly or incorrectly classified.

For example, a confusion matrix may look like:

```text
                Predicted
              Setosa  Versicolor  Virginica

Actual Setosa    10       0           0

Actual Versicolor 0       9           1

Actual Virginica 0        1           9
```

The values along the diagonal represent correctly classified samples.

In this example:

```text
Setosa      → 10 correctly classified
Versicolor  → 9 correctly classified
Virginica   → 9 correctly classified
```

The off-diagonal values represent misclassified samples.

---

## 17. Accuracy

Accuracy is calculated as:

```text
Accuracy = Correct Predictions / Total Predictions
```

It can also be expressed as:

```text
Accuracy = (TP + TN) / Total Samples
```

For a multi-class classification problem, the overall accuracy represents the proportion of all test samples that were classified correctly.

The accuracy can be calculated using:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Classification Accuracy:", accuracy)
```

---

## 18. Complete Workflow

The complete MLP classification workflow used in this assignment is:

```text
Iris Dataset
     ↓
Load Dataset
     ↓
Explore Dataset
     ↓
Check Missing Values
     ↓
Separate Features and Labels
     ↓
Train-Test Split
     ↓
Feature Normalization
     ↓
Build MLP
     ↓
Compile Model
     ↓
Train Model
     ↓
Make Predictions
     ↓
Calculate Accuracy
     ↓
Generate Confusion Matrix
     ↓
Evaluate Classification Performance
```

---

## 19. Concepts Covered

This assignment demonstrates the following Deep Learning concepts:

* Multilayer Perceptron (MLP)
* Artificial Neural Networks
* Input and hidden layers
* Dense layers
* ReLU activation function
* Softmax activation function
* Forward propagation
* Model compilation
* Adam optimizer
* Loss functions
* Model training
* Classification accuracy
* Confusion matrix
* Model evaluation

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

The Iris dataset contains **150 samples** belonging to three different iris species.

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

[Open the Assignment 02 notebook in Google Colab](https://colab.research.google.com/)

## Files

* `Assignment_02.ipynb` — Complete implementation of the MLP classification assignment.
* `README.md` — Documentation explaining the MLP architecture, preprocessing, training, accuracy evaluation, and confusion matrix.
