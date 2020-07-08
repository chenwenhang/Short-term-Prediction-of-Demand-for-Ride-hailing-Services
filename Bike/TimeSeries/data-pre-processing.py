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


df = pd.read_csv('../data/NYC-BikeShare-2015-2017-combined.csv').dropna(how='any')

# Extract information of interest from the raw data
col_of_interest = ['Trip Duration', 'Start Time', 'Start Station ID', 'End Station ID']
trips = df[col_of_interest]

# Select valid data, we think that the data
# with travel time between 60 and 7200 is valid data
trips = trips[(trips['Trip Duration'] >= 60) & (trips['Trip Duration'] <= 7200)]

# Splitting timestamp column (e.g. 2017-03-09 21:30:11) into separate date and time columns
# Add a time period column in 15 minutes, and a day of week column to indicate weekday or weekend
time_interval = 15  # in minutes, e.g. 15 minutes
trips.loc[:, 'Start Time'] = pd.DatetimeIndex(trips['Start Time'])
new_dates, new_times, new_year, new_dayofweek, new_dayofyear, new_time_period = zip(*[(d.date(),
                                                                                       d.time(),
                                                                                       d.year,
                                                                                       d.weekday(),
                                                                                       d.dayofyear,
                                                                                       time_transformation(d.time(),
                                                                                                           time_interval))
                                                                                      for d in trips['Start Time']])

# Notice: since the dataset including 2015-2017 share-bike data, we need
# to distinguish between different years, otherwise when we add up the
# number of passengers for each time period later, the data of the same day
# for three years will be added up, which is wrong apparently
trips = trips.assign(Date=new_dates, DayofWeek=new_dayofweek, DayofYear=new_dayofyear,
                     Time=new_times, TimePeriod=new_time_period, Year=new_year)

# Select only the data from 5:00am to 23:00pm
trips = trips[(trips.TimePeriod >= 5 * 60) & (trips.TimePeriod < 23 * 60)]

# Save the results to the external csv files
trips = trips[['Date', 'Time', 'Year', 'DayofYear', 'DayofWeek', 'TimePeriod', 'Start Station ID']]

# Rename column name to facilitate subsequent operations
trips.rename(columns={'Start Station ID': 'StartStationId'}, inplace=True)

# Add passenger_count=1 to facilitate subsequent operations (It makes summation easier)
trips['passenger_count'] = 1
trips.to_csv('../data/NY_bike_processed.csv', index=False)
