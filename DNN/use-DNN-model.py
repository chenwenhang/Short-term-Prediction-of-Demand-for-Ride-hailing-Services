from keras.models import load_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Set data_max and data_min
x_data_max = np.array([90, 1365, 6])
x_data_min = np.array([60, 300, 0])
y_data_max = np.array([9877])
y_data_min = np.array([419])

# Read model
model = load_model('./model/kera.h5')

# Use model
X = np.array([[84, 480, 0], [67, 570, 4]])

# Pre-processing
X = (X - x_data_min) / (x_data_max - x_data_min)
poly = PolynomialFeatures(degree=4, include_bias=True, interaction_only=False)
X = poly.fit_transform(X)

# Predict
y = model.predict(X)

# Inverse normalization
y_pred = y * (y_data_max - y_data_min) + y_data_min

print('Predict result:\n', y_pred)
