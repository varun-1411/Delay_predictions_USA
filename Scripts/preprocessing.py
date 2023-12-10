import numpy as np
from sklearn import preprocessing

from main import merged_df

# 1st method to scale
# x_scaled = preprocessing.scale(X)
#
# scalar = preprocessing.StandardScaler()
# scalar.fit(x)
# scalar.transform(x)
# scalar.transform(x_test)

# Step 2: Sort the Data by Day of the Month
merged_df.sort_values(by='DAY_OF_MONTH', inplace=True)

# Step 3: Calculate Split Points
total_rows = merged_df.shape[0]
train_split_point = int(0.7 * total_rows)
val_split_point = int(0.9 * total_rows)

# Step 4: Split the Data
train_data = merged_df.iloc[:train_split_point]
val_data = merged_df.iloc[train_split_point:val_split_point]
test_data = merged_df.iloc[val_split_point:]

# Print the size of each split
print("Train data size:", train_data.shape[0])
print("Validation data size:", val_data.shape[0])
print("Test data size:", test_data.shape[0])

# Aggregation Features
avg_dep_delay_by_origin = train_data.groupby('ORIGIN')['DEP_DELAY'].mean()
train_data['AVG_DEP_DELAY_ORIGIN'] = train_data['ORIGIN'].map(avg_dep_delay_by_origin)
train_data['SPEED'] = train_data['DISTANCE'] / train_data['AIR_TIME'] # mile per minute


numerical_features = ['CRS_DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF',
                      'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_DELAY', 'CRS_ELAPSED_TIME',
                      'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'CARRIER_DELAY',
                      'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'ORIGIN_LATITUDE',
                      'ORIGIN_LONGITUDE',
                      'ORIGIN_ELEVATION', 'ORIGIN_HourlyAltimeterSetting',
                      'ORIGIN_HourlyDewPointTemperature', 'ORIGIN_HourlyDryBulbTemperature',
                      'ORIGIN_HourlyPrecipitation', 'ORIGIN_HourlyRelativeHumidity',
                      'ORIGIN_HourlySeaLevelPressure',
                      'ORIGIN_HourlyStationPressure', 'ORIGIN_HourlyVisibility',
                      'ORIGIN_HourlyWetBulbTemperature', 'ORIGIN_HourlyWindDirection',
                      'ORIGIN_HourlyWindSpeed', 'DEST_LATITUDE', 'DEST_LONGITUDE',
                      'DEST_ELEVATION', 'DEST_HourlyAltimeterSetting',
                      'DEST_HourlyDewPointTemperature', 'DEST_HourlyDryBulbTemperature',
                      'DEST_HourlyPrecipitation', 'DEST_HourlyRelativeHumidity',
                      'DEST_HourlySeaLevelPressure',
                      'DEST_HourlyStationPressure', 'DEST_HourlyVisibility',
                      'DEST_HourlyWetBulbTemperature', 'DEST_HourlyWindDirection',
                      'DEST_HourlyWindSpeed']

# Create a StandardScaler and fit it on the training data
scaler = preprocessing.StandardScaler()
scaler.fit(train_data[numerical_features])

# Apply the same scaling to both the training and testing data
train_data[numerical_features] = scaler.transform(train_data[numerical_features])
train_data[numerical_features] = scaler.transform(val_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])
