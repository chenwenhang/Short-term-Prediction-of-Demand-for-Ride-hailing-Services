from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import time

# Make sure the data file and python code is in the same folder
start = time.time()
trips = pd.read_csv('../data/NY_bike_processed.csv')

# Check data information
trips.head()
trips.info()

# Calculate passenger trips per time period per day
trips_period = trips.groupby(['Year', 'DayofYear', 'TimePeriod', 'DayofWeek'])['passenger_count'].sum().reset_index()

# Use year 2016
trips_period = trips_period[(trips_period.Year == 2016)]

# Construct a template of times
time_interval = 15
day_start, day_end = (0, 30)  # March, days starting from January 1
time_start, time_end = (5 * 60, 23 * 60)  # 5:00-23:00 in minutes
unique_days = np.arange(day_start, day_end)
unique_time_periods = np.arange(time_start, time_end, time_interval)  # time period with interval 15 minutes
list_day, list_time = zip(*[(d, t) for d in unique_days for t in unique_time_periods])
ts_template = pd.DataFrame()
ts_template = ts_template.assign(DayofYear=list_day, TimePeriod=list_time)

# Merge the observations with the template and fill na zero
ts = ts_template.merge(trips_period, on=['DayofYear', 'TimePeriod'], how='left').fillna(0)

# Disrupt the list order
ts = shuffle(ts)

print(ts)

# Get features and results
X = ts[['DayofYear', 'TimePeriod', 'DayofWeek']]
y = ts[['passenger_count']]

# Because of the small number of features, the trained model scores low,
# so PolynomialFeatures function needs to be used to increase the number of features
poly = PolynomialFeatures(degree=4, include_bias=True, interaction_only=False)
X = poly.fit_transform(X)

print(X.shape)
print(y.shape)

# Split train_set and test_set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# Build model
ridgeCV = Ridge()

# Generate optional parameter dictionary
alpha = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
max_iter = [1000, 10000, 100000]
param_grid = dict(alpha=alpha, max_iter=max_iter)

# Use GridSearchCV to adjust super-parameters automatically
# Use all CPU cores, five times cross-validation
grid_search = GridSearchCV(ridgeCV, param_grid, n_jobs=-1, cv=5)
grid_result = grid_search.fit(X_train, y_train)

# Print best result
print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

# Print all results
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))

# Save model
model = Ridge(alpha=grid_search.best_params_['alpha'],
              max_iter=grid_search.best_params_['max_iter']).fit(X_train, y_train)
joblib.dump(model, './model/ridge.pkl')

# Print final estimate results
y_hat = model.predict(np.array(X_test))

# DataFrame to ndarray, otherwise types
# do not match and they cannot be subtracted
y_test = np.array(y_test)
error = y_hat - y_test
print("R2 score: %f" % r2_score(y_test, y_hat))

# Draw picture
# compare of demand against time of day
t = np.arange(len(X_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.savefig("./image/demand_against_time_of_day.png")
plt.show()

# error Visualization
plt.hist(error)
plt.savefig("./image/hist.png")
plt.show()
plt.scatter(y_test, error)
plt.savefig("./image/error_visualization.png")
plt.show()

# scatter plot of prediction and true values
plt.scatter(y_test, y_hat)
plt.savefig("./image/prediction_and_true.png")
plt.show()

# Code running time
end = time.time()
print(end - start)
