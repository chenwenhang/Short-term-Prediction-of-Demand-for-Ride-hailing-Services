import pandas as pd
import datetime as dt


def time_transformation(obj_dt_time, time_interval):
    # transform the datetime.time object to time period
    """
    :param obj_dt_time: datetime.time object
    :param time_interval: in minute, e.g. 15 minutes
    :return: time period in minutes index from the mid-night, e.g. 5:15 is 5*60 + 15 = 315
    """
    total_seconds = obj_dt_time.hour * 3600 + obj_dt_time.minute * 60 + obj_dt_time.second
    trunc_seconds = total_seconds // (time_interval * 60) * (time_interval * 60)
    time_period = trunc_seconds // 60

    return time_period


df = pd.read_csv('./data/yellow_tripdata_2019-03.csv').dropna(how='any')

# Extract information of interest from the raw data
col_of_interest = ['tpep_pickup_datetime', 'passenger_count', 'PULocationID', 'DOLocationID']
trips = df[col_of_interest]

# Splitting timestamp column (e.g. 2017-03-09 21:30:11) into separate date and time columns
# Add a time period column in 15 minutes, and a day of week column to indicate weekday or weekend
time_interval = 15  # in minutes, e.g. 15 minutes
trips.loc[:, 'tpep_pickup_datetime'] = pd.DatetimeIndex(trips['tpep_pickup_datetime'])
new_dates, new_times, new_dayofweek, new_dayofyear, new_time_period = zip(*[(d.date(),
                                                                             d.time(),
                                                                             d.weekday(),
                                                                             d.dayofyear,
                                                                             time_transformation(d.time(),
                                                                                                 time_interval))
                                                                            for d in trips['tpep_pickup_datetime']])

trips = trips.assign(Date=new_dates, DayofWeek=new_dayofweek, DayofYear=new_dayofyear,
                     Time=new_times, TimePeriod=new_time_period)

# Select only the data from 5:00am to 23:00pm
trips = trips[(trips.TimePeriod >= 5 * 60) & (trips.TimePeriod < 23 * 60)]

# Save the results to the external csv files
trips = trips[['Date', 'Time', 'PULocationID', 'DayofYear', 'DayofWeek', 'TimePeriod', 'passenger_count']]
trips.to_csv('./data/NY_Taxi_March_2019.csv', index=False)
