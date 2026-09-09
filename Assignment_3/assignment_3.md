# Assignment 03 — Forward Propagation and Backpropagation using TensorFlow/Keras

## Overview

This assignment introduces the implementation of **forward propagation and backpropagation** using **TensorFlow and Keras** for a classification problem.

The assignment uses the **Iris dataset**, which contains measurements of iris flowers belonging to three different species: **Setosa, Versicolor, and Virginica**.

The main purpose of this assignment is to understand how a neural network performs forward propagation to generate predictions and uses backpropagation to update its weights and minimize the loss during training.

The assignment also analyzes the effect of different **learning rates** and **numbers of epochs** on the performance of the neural network.

---

## 1. TensorFlow and Keras Setup

The assignment begins by importing the required libraries for implementing and training the neural network.

The following libraries are used:

* **TensorFlow** — used as the primary Deep Learning framework.
* **Keras** — used for defining and training the neural network.
* **NumPy** — used for numerical operations.
* **Pandas** — used for data manipulation and analysis.
* **Matplotlib** — used for visualizing model performance.
* **Scikit-learn** — used for loading the Iris dataset, preprocessing, and train-test splitting.

The implementation is performed in **Google Colab**, which provides a Python environment suitable for running TensorFlow and Keras models.

---

## 2. Loading the Dataset

The **Iris dataset** is used for the classification task.

The dataset contains **150 samples** and **4 numerical input features**.

The features are:

* **Sepal Length**
* **Sepal Width**
* **Petal Length**
* **Petal Width**

The target variable represents three different species of iris flowers:

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

The objective of the neural network is to learn the relationship between the four input features and correctly classify each sample into one of the three Iris classes.

---

## 3. Data Preprocessing

Before training the neural network, the dataset is prepared for the model.

The input features and target labels are separated:

```text
X → Input Features
y → Target Labels
```

The dataset is then divided into training and testing sets.

Feature scaling is also performed so that the input features have a similar scale.

---

## 4. Train-Test Split

The Iris dataset is divided into training and testing sets using an **80:20 split**.

```text
Training Data → 80%
Testing Data  → 20%
```

This results in:

```text
Training samples → 120
Testing samples  → 30
```

The training dataset is used for learning the model parameters, while the testing dataset is kept separate for evaluating the trained model.

A fixed random state is used to make the experiment reproducible.

---

## 5. Feature Normalization

The Iris features have different numerical ranges. Therefore, **StandardScaler** is used to standardize the input features.

The standardization formula is:

```text
z = (x - μ) / σ
```

where:

* `x` = original feature value
* `μ` = mean of the feature
* `σ` = standard deviation of the feature
* `z` = standardized feature value

The scaler is fitted using the training data and then applied to both the training and testing data.

```text
Training Data
      ↓
Fit Scaler
      ↓
Transform Training Data
      ↓
Transform Testing Data
```

This ensures that information from the testing dataset does not influence the training process.

---

## 6. Neural Network Architecture

A simple feed-forward neural network is used to demonstrate forward propagation and backpropagation.

The architecture consists of:

```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

A typical architecture used for the Iris classification task is:

```text
Input Layer       → 4 features
Hidden Layer 1    → 16 neurons, ReLU
Hidden Layer 2    → 8 neurons, ReLU
Output Layer      → 3 neurons, Softmax
```

The input layer receives the four Iris features.

The hidden layers perform transformations on the input using learned weights and biases.

The output layer contains three neurons corresponding to the three Iris classes.

The **Softmax** activation function converts the output values into class probabilities.

---

## 7. Forward Propagation

**Forward propagation** is the process through which input data moves from the input layer through the hidden layers to the output layer.

For each layer, the weighted sum is calculated as:

```text
z = Wx + b
```

where:

* `W` = weights
* `x` = input
* `b` = bias
* `z` = weighted sum

An activation function is then applied:

```text
a = f(z)
```

For the hidden layers, the **ReLU** activation function is used:

```text
ReLU(x) = max(0, x)
```

The final layer uses Softmax to produce probabilities for the three classes.

The forward propagation process can be represented as:

```text
Input Features
      ↓
Weighted Sum
      ↓
ReLU Activation
      ↓
Hidden Layer
      ↓
Weighted Sum
      ↓
ReLU Activation
      ↓
Hidden Layer
      ↓
Weighted Sum
      ↓
Softmax
      ↓
Class Probabilities
```

For example, the output could be:

```text
Class 0 → 0.02
Class 1 → 0.95
Class 2 → 0.03
```

The class with the highest probability becomes the predicted class.

---

## 8. Loss Calculation

After forward propagation, the predicted output is compared with the actual target label.

For multi-class classification, **Sparse Categorical Crossentropy** is used as the loss function.

The loss indicates how different the predicted probabilities are from the actual class.

The objective of training is to minimize this loss.

```text
Actual Label
      ↓
Compare with
      ↑
Predicted Probability
      ↓
Calculate Loss
```

A lower loss generally indicates that the model's predictions are closer to the correct labels.

---

## 9. Backpropagation

**Backpropagation** is the process used to update the weights and biases of the neural network based on the calculated error.

After calculating the loss, the gradients of the loss with respect to the model parameters are calculated.

The gradients are propagated backward through the network.

The process can be represented as:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Calculate Gradients
  ↓
Backpropagation
  ↓
Update Weights and Biases
  ↓
Next Training Step
```

TensorFlow automatically calculates these gradients using **automatic differentiation**.

The optimizer then uses the calculated gradients to update the model parameters.

---

## 10. Weight Update

The basic gradient descent weight update can be represented as:

```text
Wnew = Wold - η × ∂L/∂W
```

where:

* `W` = model weights
* `L` = loss
* `η` = learning rate
* `∂L/∂W` = gradient of the loss with respect to the weights

The learning rate determines how large each parameter update is.

A very small learning rate may result in slow learning, while a very large learning rate can make training unstable.

---

## 11. Learning Rate

The **learning rate** is one of the important hyperparameters of a neural network.

It controls how much the model's weights are changed during each optimization step.

In this assignment, multiple learning rates are tested to analyze their effect on model performance.

For example:

```text
Learning Rates:

0.001
0.01
0.1
```

The same model architecture and dataset are used while changing only the learning rate.

This allows the effect of the learning rate to be compared fairly.

---

## 12. Effect of Learning Rate

Different learning rates can produce different training behavior.

### Small Learning Rate

A small learning rate results in smaller weight updates.

```text
Small Learning Rate
        ↓
Small Parameter Updates
        ↓
Slower Learning
```

The model may require more epochs to reach good performance.

### Appropriate Learning Rate

An appropriate learning rate allows the model to learn efficiently.

```text
Appropriate Learning Rate
        ↓
Stable Weight Updates
        ↓
Faster Convergence
        ↓
Good Performance
```

### Large Learning Rate

A very large learning rate can result in large parameter updates.

```text
Large Learning Rate
        ↓
Large Parameter Updates
        ↓
Unstable Training
        ↓
Loss May Fluctuate
```

Therefore, the learning rate must be selected carefully.

---

## 13. Number of Epochs

An **epoch** represents one complete pass through the training dataset.

Different numbers of epochs are tested to analyze their effect on model performance.

For example:

```text
Epochs:

10
50
100
```

The model is trained using the same learning rate while changing the number of epochs.

---

## 14. Effect of Number of Epochs

The number of epochs determines how many times the model sees the complete training dataset.

### Too Few Epochs

If the number of epochs is too small:

```text
Few Epochs
    ↓
Insufficient Training
    ↓
Model May Underfit
```

The model may not have enough time to learn the patterns in the dataset.

### Appropriate Number of Epochs

With a suitable number of epochs:

```text
More Training
     ↓
Better Learning
     ↓
Improved Performance
```

### Too Many Epochs

Training for too many epochs can result in overfitting.

```text
Excessive Epochs
      ↓
Model Learns Training Data Too Closely
      ↓
Possible Overfitting
```

Therefore, both training and validation performance should be observed.

---

## 15. Comparing Different Learning Rates

The performance of different learning rates can be compared using a table.

For example:

| Learning Rate | Epochs | Training Accuracy | Testing Accuracy |
| ------------- | -----: | ----------------: | ---------------: |
| 0.001         |     50 |      Model Result |     Model Result |
| 0.01          |     50 |      Model Result |     Model Result |
| 0.1           |     50 |      Model Result |     Model Result |

The actual values should be filled using the results obtained after running the notebook.

The learning rate producing stable training and better test performance can then be identified.

---

## 16. Comparing Different Numbers of Epochs

Similarly, the effect of different numbers of epochs can be analyzed.

| Learning Rate | Epochs | Training Accuracy | Testing Accuracy |
| ------------- | -----: | ----------------: | ---------------: |
| 0.01          |     10 |      Model Result |     Model Result |
| 0.01          |     50 |      Model Result |     Model Result |
| 0.01          |    100 |      Model Result |     Model Result |

This comparison helps determine whether increasing the number of epochs improves the model's performance.

---

## 17. Training and Validation Accuracy

The training history can be used to visualize model performance across epochs.

```python
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.show()
```

The graph can be used to observe:

* Learning progress
* Convergence
* Underfitting
* Possible overfitting
* Effect of training duration

---

## 18. Training and Validation Loss

Loss can also be visualized during training.

```python
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.show()
```

A decreasing training loss generally indicates that the model is learning from the training data.

If validation loss starts increasing while training loss continues decreasing, it may indicate overfitting.

---

## 19. Model Evaluation

After training, the model is evaluated using the testing dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

The test accuracy represents the proportion of test samples that were correctly classified by the neural network.

The exact performance depends on the selected learning rate, number of epochs, model architecture, and train-test split.

---

## 20. Complete Training Process

The complete forward propagation and backpropagation training process can be represented as:

```text
Input Data
    ↓
Forward Propagation
    ↓
Calculate Predictions
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Calculate Gradients
    ↓
Update Weights
    ↓
Next Epoch
    ↓
Repeat
    ↓
Trained Model
```

TensorFlow and Keras handle the gradient calculation and parameter updates through automatic differentiation and the selected optimizer.

---

## 21. Experimental Analysis

The main experiment in this assignment is to study how changing the **learning rate** and **number of epochs** affects model performance.

The following factors are analyzed:

### Learning Rate

```text
0.001
0.01
0.1
```

### Number of Epochs

```text
10
50
100
```

For each experiment, the model's training and testing performance can be recorded.

The results can then be compared to determine:

* Which learning rate provides stable training
* Which learning rate converges faster
* Whether increasing epochs improves accuracy
* Whether excessive training causes overfitting
* Which combination gives the best test performance

---

## 22. Concepts Covered

This assignment demonstrates the following Deep Learning concepts:

* Neural Networks
* Multilayer Perceptron
* Forward Propagation
* Weighted Sum
* Activation Functions
* ReLU
* Softmax
* Loss Calculation
* Cross-Entropy Loss
* Backpropagation
* Gradients
* Gradient Descent
* Learning Rate
* Epochs
* Model Training
* Model Evaluation
* Training and Validation Accuracy
* Training and Validation Loss
* Hyperparameter Analysis

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

The Iris dataset contains **150 samples** belonging to three different species of iris flowers.

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

[Open the Assignment 03 notebook in Google Colab](https://colab.research.google.com/)

## Files

* `Assignment_03.ipynb` — Complete implementation of forward propagation, backpropagation, and learning-rate/epoch experiments.
* `README.md` — Documentation explaining the neural network, forward propagation, backpropagation, learning rate, epochs, and model performance analysis.
