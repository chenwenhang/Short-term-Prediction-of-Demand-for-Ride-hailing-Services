from sklearn.preprocessing import PolynomialFeatures
import joblib
import numpy as np

# Read model
model = joblib.load('./model/ridge.pkl')

# Use model
test = np.array([[84, 480, 0], [67, 570, 4]])
poly = PolynomialFeatures(degree=4, include_bias=True, interaction_only=False)
test = poly.fit_transform(test)

print('Predict result:\n', model.predict(test))
