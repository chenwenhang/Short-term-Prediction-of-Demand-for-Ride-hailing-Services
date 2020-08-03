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

# Make sure the data file and python code is in the same folder
start = time.time()
trips = pd.read_csv('../data/NY_Taxi_March_2019.csv')

# Check data information
trips.head()
trips.info()

# Calculate passenger trips per time period per day
trips['boardings'] = 1
trips_period = trips.groupby(['DayofYear', 'TimePeriod', 'DayofWeek'])['boardings'].sum().reset_index()
trips_period = trips_period.groupby(['TimePeriod'])['boardings'].mean().reset_index()

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
ts = ts_template.merge(trips_period, on=['TimePeriod'], how='left').fillna(0)

# Time Series need to sort data records by time
ts.sort_values(['TimePeriod'], inplace=True, ascending=True)

x = ts['TimePeriod']
y = ts['boardings']
plt.scatter(x, y, s=5)
plt.title("Boardings Change", fontsize=22)
plt.xlabel("TimePeriod", fontsize=12)
plt.ylabel("Boardings", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=10)
xticks = [i for i in range(300, 1365, 200)]
xticklabes = [i for i in range(300, 1365, 200)]
plt.xticks(xticks, xticklabes, size=14, color='grey')
plt.show()

print(ts)
