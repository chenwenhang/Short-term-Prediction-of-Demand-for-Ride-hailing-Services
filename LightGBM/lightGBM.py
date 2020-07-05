from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMClassifier
import joblib
import numpy as np
import pandas as pd
import time

# Make sure the data file and python code is in the same folder
start = time.time()
trips = pd.read_csv('./data/NY_Taxi_March_2019.csv')

# Check data information
trips.head()
trips.info()

# Calculate passenger trips per time period per day
trips_period = trips.groupby(['DayofYear', 'TimePeriod', 'DayofWeek'])['passenger_count'].sum().reset_index()

# Construct a template of times
time_interval = 15
day_start, day_end = (60, 91)  # March, days starting from January 1
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

# Build model12
lightGBM = LGBMClassifier()

# Generate optional parameter dictionary
num_leaves = [31, 50, 100]
max_depth = [-1]
learning_rate = [0.01]
boosting = ["dart"]
objective = ['regression']

# Use GPU to accelerate training model
# device = ['gpu']
# gpu_platform_id = [1]
# gpu_device_id = [0]
# param_grid = dict(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=learning_rate,
#                   device=device, gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id)

param_grid = dict(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
                  objective=objective, boosting=boosting)

# Use GridSearchCV to adjust super-parameters automatically
# Use all CPU cores, four times cross-validation
grid_search = GridSearchCV(lightGBM, param_grid, n_jobs=-1, cv=4)
grid_result = grid_search.fit(X_train, y_train)

# Print best result
print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

# Print all results
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))

# Save model
model = LGBMClassifier(num_leaves=grid_search.best_params_['num_leaves'],
                       max_depth=grid_search.best_params_['max_depth'],
                       learning_rate=grid_search.best_params_['learning_rate'],
                       boosting=grid_search.best_params_['boosting'],
                       objective=grid_search.best_params_['objective']).fit(X_train, y_train)
joblib.dump(model, './model/lightGBM.pkl')

# Code running time
end = time.time()
print(end - start)
