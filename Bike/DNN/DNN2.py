# %%

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import time

# Make sure the data file and python code is in the same folder
start = time.time()
path = '../data/'
trips = pd.read_csv(path + 'NY_bike_processed.csv')
# Check data information
trips.head()
# I: Calculate passenger trips per time period per day
trips_period = trips.groupby(['Year', 'DayofYear', 'TimePeriod'])['passenger_count'].sum().reset_index()
trips_period.head()
# Use year 2016
trips_period = trips_period[(trips_period.Year == 2016)]
# step 1: Construct a template of times
time_interval = 15  # 15 min
day_start, day_end = (0, 365)  # days starting from January 1
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
ts['Year'] == 2016
ts.head()
# Get features and results
X = ts[['DayofYear', 'TimePeriod', 'Year']]
y = ts[['passenger_count']]

# Normalized
n_sample = X.shape[0]
scalerX, scalerY = MinMaxScaler(), MinMaxScaler()
scalerX.fit(X)
scalerY.fit(y.values.reshape(n_sample, 1))
X = scalerX.transform(X)
y = scalerY.transform(y.values.reshape(n_sample, 1))

# Because of the small number of features, the trained model scores low,
# so PolynomialFeatures function needs to be used to increase the number of features
poly = PolynomialFeatures(degree=4, include_bias=True, interaction_only=False)
X = poly.fit_transform(X)
print(X.shape)
print(y.shape)
# Split train_set and test_set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)

# %%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# %%

n_feature = X.shape[1]
model = Sequential()
model.add(Dense(4, input_dim=n_feature, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=1000, verbose=0)

# %%

# Predict
y_hat = model.predict(np.array(X_test))
error = y_hat - y_test
print("R2 score: %f" % r2_score(y_test, y_hat))

# Save model
model.save('./model/kera.h5')

# Draw picture
# compare of demand against time of day
t = np.arange(len(X_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.savefig("./image/demand_against_time_of_day.png")
plt.show()

# %%

# error Visualization
plt.hist(error)
plt.savefig("./image/hist.png")
plt.show()
plt.scatter(y_test, error)
plt.xlabel('y_test')
plt.ylabel('error')
plt.savefig("./image/error_visualization.png")
plt.show()

# %%

# scatter plot of prediction and true values
plt.scatter(y_test, y_hat)
plt.xlabel('y_test')
plt.ylabel('y_hat')
plt.savefig("./image/prediction_and_true.png")
plt.show()

# Code running time
end = time.time()
print(end - start)

# %%


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print(f"Mean Squared Error(MSE)：{mean_squared_error(y_hat, y_test)}")
print(f"Root Mean Squared Error(RMSE)：{np.sqrt(mean_squared_error(y_hat, y_test))}")
print(f"test set R^2：{r2_score(y_test, y_hat)}")

# %%


# %%


# %%
