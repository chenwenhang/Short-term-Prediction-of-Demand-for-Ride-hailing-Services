import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import joblib
import time

# Make sure the data file and python code is in the same folder
start = time.time()
trips = pd.read_csv('../data/NY_bike_processed.csv')

# Check data information
trips.head()
trips.info()

# I: Calculate passenger trips per time period per day
trips_period = trips.groupby(['Year', 'DayofYear', 'TimePeriod'])['passenger_count'].sum().reset_index()

# Use year 2016
trips_period = trips_period[(trips_period.Year == 2016)]

# II: construct time series
# step 1: Construct a template of times
time_interval = 15  # 15 min
day_start, day_end = (0, 30)  # days starting from January 1
time_start, time_end = (5 * 60, 23 * 60)  # 5:00-23:00 in minutes
unique_days = np.arange(day_start, day_end)
unique_time_periods = np.arange(time_start, time_end, time_interval)  # time period with interval 15 minutes
list_day, list_time = zip(*[(d, t) for d in unique_days for t in unique_time_periods])
ts_template = pd.DataFrame()
ts_template = ts_template.assign(DayofYear=list_day, TimePeriod=list_time)

# step 2: Merge the observations with the template and fill na zero
ts = ts_template.merge(trips_period, on=['DayofYear', 'TimePeriod'], how='left').fillna(0)

# Time Series need to sort data records by time
ts.sort_values(['DayofYear', 'TimePeriod'], inplace=True, ascending=False)

# step 3: Create columns for time series of pax counts with time lags (t-1, t-2, t-3, etc.)
ts['T'] = ts['passenger_count']
ts['T_1'] = ts['T'].shift(-1)
ts['T_2'] = ts['T'].shift(-2)
ts['T_3'] = ts['T'].shift(-3)

# step 4: Delete the record with day transition time periods
del_list = np.arange(time_start, time_start + time_interval * 3, time_interval)
ts = ts[~ts.TimePeriod.isin(del_list)]

# step 5: Finalize the data, and split training and testing data
ts_train = ts[ts.DayofYear <= 20]
ts_test = ts[ts.DayofYear > 20]
ts = ts[['T', 'T_1', 'T_2', 'T_3']]

""" ********************************* DATA EXPLORATION ********************************************"""
# I. Descriptive statistics
desc_stat = ts.describe()

# II. Distribution and scatter plots
sns.pairplot(ts)  # pair plot
plt.savefig("./image/pairplot.png")
plt.show()
ts_corr = ts.corr()  # correlation matrix (linear relationship)

# III. Correlation and visualization
sns.heatmap(ts_corr)
plt.savefig("./image/heatmap.png")
plt.show()

""" ********************************* Linear Regression Model ********************************************"""
# Training a Linear Regression Model
# step 1： Regression model demand(T) = a1 * demand(T-1) + a2 * demand(T-2) + a3 * demand(T-3)
X_train = ts_train[['T_1', 'T_2', 'T_3']]
y_train = ts_train['T']

X_test = ts_test[['T_1', 'T_2', 'T_3']]
y_test = ts_test['T']

# step 2: Train linear regression model using the training data
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train)
results = model.fit()

# save model
joblib.dump(model, './model/statsmodels.pkl')

# step 3: Assess performance
print(results.summary())

# step 4: Check assumption of i.i.d
X_test = sm.add_constant(X_test)
predictions = results.predict(X_test)
error = predictions - y_test
print("R2 score: %f" % r2_score(y_test, predictions))

plt.hist(error)
plt.savefig("./image/hist.png")
plt.show()
plt.scatter(y_test, error)
plt.savefig("./image/error_visualization.png")
plt.show()

# step 5: Use the regression for prediction
# scatter plot of prediction and true values
plt.scatter(y_test, predictions)
plt.savefig("./image/prediction_and_true.png")
plt.show()

# compare of demand against time of day
x = range(len(y_test))
plt.plot(x, y_test)
plt.plot(x, predictions)
plt.savefig("./image/demand_against_time_of_day.png")
plt.show()

# Code running time
end = time.time()
print(end - start)
