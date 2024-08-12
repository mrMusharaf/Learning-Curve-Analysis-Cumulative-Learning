# **Cumulative Learning Process Implementation**

This repository provides an implementation of a process designed to train models on cumulative data subsets and evaluate their performance. The approach is broken down into three key steps: dividing the data into subsets, training models on cumulative data, and plotting validation performance.

## **Step 1: Divide Training Data into "k" Subsets**

In this step, the training data is divided into several subsets. Although the code uses `scikit-learn`'s `learning_curve` function, which automatically handles data splitting, the approach can be manually implemented for more control.

### Code:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's say we want to divide into 7 subsets
k = 7
train_sizes = np.linspace(0.1, 1.0, k)
```

## **Step 2: Train Models on Cumulative Data and Evaluate on Validation Data**

Here, models are trained on increasingly larger portions of the training data, and their performance is evaluated on a separate validation set. This helps in understanding how model accuracy improves with more data.

### Code:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the model
model = RandomForestClassifier(random_state=42)

train_scores = []
val_scores = []

# Train and evaluate the model on cumulative data
for size in train_sizes:
    X_train_subset = X_train[:int(size * len(X_train))]
    y_train_subset = y_train[:int(size * len(y_train))]
    
    # Train the model on the subset
    model.fit(X_train_subset, y_train_subset)
    
    # Evaluate on the training and validation data
    train_scores.append(accuracy_score(y_train_subset, model.predict(X_train_subset)))
    val_scores.append(accuracy_score(y_val, model.predict(X_val)))
```

## **Step 3: Plot Validation Performance**

Finally, the training and validation scores are plotted to visualize the model's performance as more training data is used. This plot helps in identifying whether adding more data could improve model accuracy.

### Code:
```python
# Plot the learning curve
plt.figure()
plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
plt.plot(train_sizes, val_scores, 'o-', color="g", label="Validation score")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.grid()
plt.show()
```

## **Summary**

1. **Step 1:** The data is divided into subsets using `np.linspace` to create varying training sizes.
2. **Step 2:** Models are trained on each subset, and their performance is evaluated on validation data.
3. **Step 3:** The training and validation scores are plotted to determine if additional data could enhance the model's performance.

This repository provides a clear and structured approach to cumulative learning, making it easier to understand and implement.
