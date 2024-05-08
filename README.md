# Numpy-Linear-Regression

This repository contains an implementation of a Linear Regression model using Numpy, developed by Shehryar Sohail and Abdulwadood Waseem. The project is designed for educational purposes, focusing on regression problems and applying machine learning techniques to analyze and work with provided datasets.

## Features

- **Linear Regression Model**: Implements a linear model with coefficients to minimize the residual sum of squares between observed targets in the dataset and the targets predicted by the linear approximation.
- **Regularization**: Supports Lasso and Ridge regularization techniques to prevent overfitting.
- **Customizable Parameters**: Allows for customization of epochs, learning rate, threshold for early stopping, alpha for regularization weight, and mode selection (Lasso/Ridge).
- **Validation Data**: Optionally splits training data into training and validation sets for model evaluation.
- **Feature Scaling**: Includes a feature scaling option using Min-Max normalization.
- **Epoch Tracking**: Displays progress with each epoch, showing loss at each iteration, and supports early stopping based on a threshold.

## Getting Started

### Prerequisites

- Python 3.x
- Numpy

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Abdulwadood39/Numpy-Linear-Regression.git
   ```
2. Navigate to the project directory:
   ```
   cd Numpy-Linear-Regression
   ```
3. Ensure you have Numpy installed. If not, install it using pip:
   ```
   pip install numpy
   ```

### Usage

1. Import the `LinearRegression` class from the `np_LinearRegression.py` file.
2. Instantiate the class with desired parameters.
3. Use the `fit` method to train the model on your dataset.
4. Use the `predict` method to make predictions with the trained model.
5. Access the model's coefficients using the `coef_` method.

### Example

```python
from np_LinearRegression import LinearRegression
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([2, 4, 6])

# Instantiate the model
model = LinearRegression(epochs=100, learning_rate=0.01, mode='Ridge', alpha=0.1)

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Access model coefficients
coefficients = model.coef_()

print("Predictions:", predictions)
print("Coefficients:", coefficients)
```

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes or additions.

## Contact

For any questions or inquiries, please contact the project maintainer at [abdulwadoodwaseem@gmail.com](mailto:abdulwadoodwaseem@gmail.com).
