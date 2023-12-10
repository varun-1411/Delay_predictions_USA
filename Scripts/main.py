import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# weather dataset

# weather_data_1 = pd.read_csv(r"3402617.csv")
# weather_data_2 = pd.read_csv(r"3402618.csv")
weather_data_3 = pd.read_csv(r"3402619.csv")
weather_data_4 = pd.read_csv(r"3402620.csv")
weather_data_5 = pd.read_csv(r"3402621.csv")
weather_data_6 = pd.read_csv(r"3402622.csv")
# weather_data_7 = pd.read_csv(r"3402627.csv")
weather_data = pd.concat(
    [weather_data_3, weather_data_4, weather_data_5, weather_data_6],
    axis=0)

# airport schedule dataset

airport_schedule = pd.read_csv(r"T_ONTIME_REPORTING.csv")
airport_schedule['CANCELLED'] = airport_schedule['CANCELLED'].astype('float')
airport_schedule = airport_schedule[airport_schedule['CANCELLED'] == 0.00000]
airport_schedule['OP_CARRIER_FL_NUM'].value_counts()
# a = airport_schedule[airport_schedule['OP_CARRIER_FL_NUM'] == 672]
airport_schedule['hour_bins_dep'] = airport_schedule['DEP_TIME_BLK'].str[:2].astype(int) + 1


# Daily variation
daily_flights = airport_schedule.groupby(['DAY_OF_MONTH','DAY_OF_WEEK','category_airport']).size().reset_index(name='Flights_dep_daily')
daily_flights['daily_flights'] = daily_flights.groupby(['DAY_OF_WEEK','category_airport'])['Flights_dep_daily'].transform('mean')
daily_flight = daily_flights[daily_flights['category_airport'] == 4 ]
plt.bar(daily_flight['DAY_OF_WEEK'], daily_flight['daily_flights'].to_numpy())
plt.xlabel('DAY OF WEEK')
plt.ylabel("Number of flights")
plt.title('Category 4 weekly variation')
plt.show()
airport_schedule = airport_schedule.merge(pd.DataFrame(daily_flights))
# Day of week
day_of_week_flights = airport_schedule.groupby(['DAY_OF_WEEK','category_airport']).size().reset_index(name='Flights_dep_weekly')
day_of_week_flight = day_of_week_flights[day_of_week_flights['category_airport'] == 1]
plt.bar(day_of_week_flight['DAY_OF_WEEK'], day_of_week_flight['Flights_dep_weekly'].to_numpy())
plt.xlabel('DAY_OF_WEEK')
plt.ylabel("Number of flights")
plt.title('Category 1 weekly variation')
plt.show()

# hourly
# a = airport_schedule[airport_schedule['ORIGIN'] == 'JFK']
grp_data = airport_schedule.groupby(['DEP_TIME_BLK', 'DAY_OF_MONTH','category_airport']).size().reset_index(name='Flights_dep')
hourly_variation = grp_data.groupby(['DEP_TIME_BLK','category_airport'])['Flights_dep'].mean().reset_index(name='Flights_dep_hourly')
# off_peak = grp_data[grp_data['DEP_TIME_BLK'] == '0001-0559']

hourly_variation_1 = hourly_variation[hourly_variation['category_airport'] == 4]
plt.bar(hourly_variation_1['DEP_TIME_BLK'], hourly_variation_1['Flights_dep_hourly'].to_numpy())
# airport_schedule = airport_schedule.merge(pd.DataFrame(hourly_variation))
x_values =hourly_variation_1['DEP_TIME_BLK'].tolist()
# Rename the x-axis values
new_x_values = [i+6 for i in range(len(x_values))]

# Update the plot with the renamed x-axis values
plt.xticks(x_values, new_x_values)
plt.xlabel('Hour')
plt.ylabel("Number of flights")
plt.xticks(rotation = 45)
plt.show()

hourly_delay = airport_schedule.groupby(['DEP_TIME_BLK','category_airport'])['DEP_DELAY_NEW'].mean().reset_index(name='Flights_del_hourly')
hour_delay = hourly_delay[hourly_delay['category_airport'] == 4]
plt.bar(hour_delay['DEP_TIME_BLK'], hour_delay['Flights_del_hourly'].to_numpy())
# x_values =hourly_delay['DEP_TIME_BLK'].tolist().unique()
new_x_values = [i+6 for i in range(len(x_values))]

# Update the plot with the renamed x-axis values
plt.xticks(x_values, new_x_values)
plt.xlabel('Hour')
plt.ylabel("Delay")
plt.xticks(rotation = 45)
plt.show()




# Airport variation

airport_var = airport_schedule.groupby(['ORIGIN']).size().reset_index(name='Flights_dep')
# airport_var = grp_data.groupby('DEP_TIME_BLK')['Flights_dep'].mean().reset_index(name='Flights_dep_hourly')

bins_edges = [0,200,800,8000,float('inf')]
labels = [1,2,3,4]
# take category_airport under dataset
airport_var['category_airport'] = pd.cut(airport_var['Flights_dep'], bins = bins_edges, labels = labels, right = False)
airport_schedule = airport_schedule.merge(pd.DataFrame(airport_var))
airport_var = airport_var.groupby('category_airport')['Flights_dep'].mean().reset_index(name='Flights_dep_avg')
# off_peak = grp_data[grp_data['DEP_TIME_BLK'] == '0001-0559']\

a = airport_var.groupby(['category_airport']).size().reset_index(name = 'x')


plt.bar(airport_var['category_airport'], airport_var['Flights_dep_avg'].to_numpy())
plt.xlabel('Airport')
plt.ylabel("Flights_dep")
plt.xticks(rotation = 45)
plt.show()

# Flight Type

flight_type_var = airport_schedule.groupby(['OP_CARRIER_FL_NUM','OP_UNIQUE_CARRIER','DAY_OF_MONTH']).size().reset_index(name='Freq')
merged = airport_schedule.merge(pd.DataFrame(flight_type_var).fillna(0))
# b = airport_schedule.where((airport_schedule['OP_CARRIER_FL_NUM'] == 490) & (airport_schedule['DAY_OF_MONTH'] == 2))


# tail_number variation
# add tail_number_variation
tail_variation = airport_schedule.groupby(['TAIL_NUM','DAY_OF_MONTH']).size().reset_index(name='Freq')
new_data = airport_schedule.merge(pd.DataFrame(tail_variation))
new_data=new_data.sort_values(by = 'DEP_TIME_BLK')
new_data = new_data[(new_data['Freq']== 12)  & (new_data["TAIL_NUM"] == 'N492HA')  & (new_data['DAY_OF_MONTH'] == 7) ]
#

plt.bar(new_data['DEP_TIME_BLK'], new_data['DEP_DELAY_NEW'].to_numpy())
plt.xlabel('Airport')
plt.ylabel("Flights_delay")
plt.xticks(rotation = 27)
plt.show()

n = new_data.groupby(['Freq','DEP_TIME_BLK'])['DEP_DELAY_NEW'].mean().reset_index(name='Flights_delay_hourly')
n1 = n[n['Freq'] ==3]
plt.bar(n1['DEP_TIME_BLK'], n1['Flights_delay_hourly'].to_numpy())
new_x_values = [i+6 for i in range(len(x_values))]
# # Update the plot with the renamed x-axis values
plt.xticks(x_values, new_x_values)
plt.xlabel('Time')
plt.ylabel("Flights_delay")
plt.title('Delay vs Freq(2)')
plt.xticks(rotation = 27)
plt.show()



# Taxi out time

# a = airport_schedule[airport_schedule['DAY_OF_MONTH'] ==1]
taxi_out_var = pd.DataFrame()
taxi_out_var['DEP_TIME_BLK'] = airport_schedule['DEP_TIME_BLK']
taxi_out_var['ORIGIN'] = airport_schedule['ORIGIN']
taxi_out_var['TAXI_OUT'] = airport_schedule['TAXI_OUT']
taxi_out_var['category_airport'] = airport_schedule['category_airport']
taxi_out_var['Avg_taxi_out'] = taxi_out_var.groupby(['DEP_TIME_BLK','category_airport'])['TAXI_OUT'].transform('mean')
taxi_out_var = taxi_out_var.sort_values(by = 'DEP_TIME_BLK')
taxi_out_var = taxi_out_var.drop_duplicates(subset = ['Avg_taxi_out','DEP_TIME_BLK','category_airport'])

var = taxi_out_var[taxi_out_var['category_airport'] == 1]
plt.bar(var['DEP_TIME_BLK'], var['Avg_taxi_out'].to_numpy())
x_values =var['DEP_TIME_BLK'].tolist()
# Rename the x-axis values
new_x_values = [i+6 for i in range(len(x_values))]

# Update the plot with the renamed x-axis values
plt.xticks(x_values, new_x_values)
plt.xlabel('Time')
plt.ylabel("Taxi_out")
plt.title('Category 1 vs Taxi out variation')
plt.xticks(rotation = 45)
plt.show()

# merged = airport_schedule.merge(pd.DataFrame(flight_type_var).fillna(0))

# Delays analysis

delay = airport_schedule.groupby(['DEP_TIME_BLK'])['DEP_DELAY_NEW'].mean().reset_index(name = 'DELAY' )
# del1 = delay[delay['category_airport'] == 4]
x_values =var['DEP_TIME_BLK'].tolist()
#
new_x_values = [i+6 for i in range(len(x_values))]

# Update the plot with the renamed x-axis values
plt.bar(delay['DEP_TIME_BLK'], delay['DELAY'].to_numpy())
# plt.xticks(x_values, new_x_values)

plt.xlabel('TIME')
plt.ylabel("Flights_delay")
plt.xticks(rotation = 27)
plt.show()
















origins = airport_schedule["ORIGIN"].value_counts()

cities = airport_schedule["ORIGIN"].value_counts()
top_five_cities = cities.nlargest(4).index.to_list()
# print(top_five_cities)
# dest_cities = airport_schedule["DEST"].value_counts()
# weather_cities = weather_data["NAME"].unique()
five_largest_airports = airport_schedule[airport_schedule["ORIGIN"].isin(top_five_cities)]
five_largest_airports = five_largest_airports[five_largest_airports["DEST"].isin(top_five_cities)]
five_largest_airports = five_largest_airports.dropna(subset=['DEP_DELAY'])
# airport_id = five_largest_airports["ORIGIN"].unique()

weather_data['timestamp'] = pd.to_datetime(weather_data['DATE'])
weather_data['date'] = weather_data['timestamp'].dt.date
weather_data['time'] = weather_data['timestamp'].dt.time
weather_data.drop(columns=["DATE"], inplace=True)
weather_data.drop(
    columns=["STATION", 'HourlyWindGustSpeed', 'REPORT_TYPE', 'DYTS', 'REM', 'SOURCE', 'HourlyPresentWeatherType',
             'HourlyPressureChange', 'HourlyPressureTendency'], inplace=True)

# weather data bins

weather_data['hour_bins'] = pd.cut(weather_data['timestamp'].dt.hour + weather_data['timestamp'].dt.minute / 60,
                                   bins=range(0, 25), labels=range(1, 25))

weather_data = weather_data.dropna(subset=['hour_bins', 'HourlyWindSpeed'])
weather_data['HourlySeaLevelPressure'] = weather_data['HourlySeaLevelPressure'].interpolate()
weather_data['HourlySeaLevelPressure'] = weather_data['HourlySeaLevelPressure'].fillna(
    weather_data['HourlySeaLevelPressure'].mean())
weather_data['HourlyPrecipitation'] = weather_data['HourlyPrecipitation'].interpolate()
weather_data['HourlyPrecipitation'] = weather_data['HourlyPrecipitation'].fillna(0)
weather_data['HourlyPrecipitation'] = np.where(weather_data['HourlyPrecipitation'] == 'T', 0,
                                               weather_data['HourlyPrecipitation'])

# only one bin per station

weather_data.drop_duplicates(subset=['NAME', 'hour_bins', 'date'], inplace=True)

# Separation of weather data in origin and destination

origin_weather_data = weather_data.add_prefix('ORIGIN_')
dest_weather_data = weather_data.add_prefix('DEST_')

# flight date
five_largest_airports['timestamp'] = pd.to_datetime(five_largest_airports['FL_DATE'])
five_largest_airports['date'] = five_largest_airports['timestamp'].dt.date

# Time bins in airport schedule

five_largest_airports['hour_bins_dep'] = five_largest_airports['DEP_TIME_BLK'].str[:2].astype(int) + 1
five_largest_airports['hour_bins_arr'] = five_largest_airports['ARR_TIME_BLK'].str[:2].astype(int) + 1

# arrival airport name matching

five_largest_airports['ORIGIN'] = np.where(five_largest_airports['ORIGIN'] == 'DFW', 'DAL FTW WSCMO AIRPORT, TX US',
                                           five_largest_airports['ORIGIN'])
# five_largest_airports['ORIGIN'] = np.where(five_largest_airports['ORIGIN'] == 'MDW', 'CHICAGO MIDWAY AIRPORT, IL US',
#                                            five_largest_airports['ORIGIN'])
# five_largest_airports['ORIGIN'] = np.where(five_largest_airports['ORIGIN'] == 'LGA', 'LAGUARDIA AIRPORT, NY US',
#                                            # five_largest_airports['ORIGIN'])
five_largest_airports['ORIGIN'] = np.where(five_largest_airports['ORIGIN'] == 'ORD', 'CHICAGO OHARE INTERNATIONAL '
                                                                                     'AIRPORT, IL US',
                                           five_largest_airports['ORIGIN'])
five_largest_airports['ORIGIN'] = np.where(five_largest_airports['ORIGIN'] == 'DEN', 'DENVER CENTENNIAL AIRPORT, CO US',
                                           five_largest_airports['ORIGIN'])
five_largest_airports['ORIGIN'] = np.where(five_largest_airports['ORIGIN'] == 'ATL', 'ATLANTA HARTSFIELD JACKSON '
                                                                                     'INTERNATIONAL AIRPORT, GA US',
                                           five_largest_airports['ORIGIN'])
# five_largest_airports['ORIGIN'] = np.where(five_largest_airports['ORIGIN'] == 'JFK', 'JFK INTERNATIONAL AIRPORT, NY US',
#                                            five_largest_airports['ORIGIN'])

# destination airport name matching

five_largest_airports['DEST'] = np.where(five_largest_airports['DEST'] == 'DFW', 'DAL FTW WSCMO AIRPORT, TX US',
                                         five_largest_airports['DEST'])
# five_largest_airports['DEST'] = np.where(five_largest_airports['DEST'] == 'MDW', 'CHICAGO MIDWAY AIRPORT, IL US',
#                                          five_largest_airports['DEST'])
# five_largest_airports['DEST'] = np.where(five_largest_airports['DEST'] == 'LGA', 'LAGUARDIA AIRPORT, NY US',
#                                          five_largest_airports['DEST'])
five_largest_airports['DEST'] = np.where(five_largest_airports['DEST'] == 'ORD', 'CHICAGO OHARE INTERNATIONAL '
                                                                                 'AIRPORT, IL US',
                                         five_largest_airports['DEST'])
five_largest_airports['DEST'] = np.where(five_largest_airports['DEST'] == 'DEN', 'DENVER CENTENNIAL AIRPORT, CO US',
                                         five_largest_airports['DEST'])
five_largest_airports['DEST'] = np.where(five_largest_airports['DEST'] == 'ATL', 'ATLANTA HARTSFIELD JACKSON '
                                                                                 'INTERNATIONAL AIRPORT, GA US',
                                         five_largest_airports['DEST'])
# five_largest_airports['DEST'] = np.where(five_largest_airports['DEST'] == 'JFK', 'JFK INTERNATIONAL AIRPORT, NY US',
#                                          five_largest_airports['DEST'])

# data merge
merged_df = five_largest_airports.merge(origin_weather_data, left_on=['date', 'ORIGIN', 'hour_bins_dep'],
                                        right_on=['ORIGIN_date', 'ORIGIN_NAME', 'ORIGIN_hour_bins'], how='left')
# merged_df.rename(columns = {'NAME':'ORG_NAME','LATITUDE':'ORG_LATITUDE','LONGITUDE':'ORG_LONGITUDE','ELEVATION'})
merged_df = merged_df.merge(dest_weather_data, left_on=['date', 'ORIGIN', 'hour_bins_dep'],
                            right_on=['DEST_date', 'DEST_NAME', 'DEST_hour_bins'], how='left')

# print(weather_cities)
# print(cities)

# Drop redundant columns
merged_df.drop(columns=['ORIGIN_NAME', "DEST_NAME", "FL_DATE", 'timestamp', 'date', 'ORIGIN_SOURCE.1', 'DEST_SOURCE.1',
                        'ORIGIN_timestamp', 'DEST_timestamp'], inplace=True)
merged_df.drop(columns=['DEST_time', 'DEST_hour_bins', 'DEST_date', 'ORIGIN_time', 'ORIGIN_hour_bins', 'ORIGIN_date',
                        'ARR_TIME_BLK', 'DEP_TIME_BLK', "CANCELLED", 'DIVERTED', 'CANCELLATION_CODE'], inplace=True)

# Fill empty values of delays
merged_df['CARRIER_DELAY'] = merged_df['CARRIER_DELAY'].fillna(0)
merged_df['WEATHER_DELAY'] = merged_df['WEATHER_DELAY'].fillna(0)
merged_df['NAS_DELAY'] = merged_df['NAS_DELAY'].fillna(0)
merged_df['SECURITY_DELAY'] = merged_df['SECURITY_DELAY'].fillna(0)
merged_df['LATE_AIRCRAFT_DELAY'] = merged_df['LATE_AIRCRAFT_DELAY'].fillna(0)
merged_df.dropna(inplace=True)


def convert_time_to_min(time_str):
    time_str = str(time_str).zfill(4)

    hours = int(time_str[:2])
    mins = int(time_str[2:])
    total_mins = hours * 60 + mins
    return total_mins


merged_df['CRS_DEP_TIME'] = merged_df['CRS_DEP_TIME'].apply(convert_time_to_min)
merged_df['CRS_ARR_TIME'] = merged_df['CRS_ARR_TIME'].apply(convert_time_to_min)


def convert_time(time_float):
    time_str = str(time_float)
    hours = int(time_str[:-8])
    mins = int(time_str[-8:-6])
    total_mins = hours * 60 + mins

    return total_mins


# print(convert_time(853.00000))
merged_df['WHEELS_ON'] = merged_df['WHEELS_ON'].apply(convert_time)
merged_df['WHEELS_OFF'] = merged_df['WHEELS_OFF'].apply(convert_time)
# print(merged_df['WHEELS_ON'].dtype)
print(merged_df.dtypes)

merged_df['ORIGIN_HourlyDewPointTemperature'].unique()


# convert strings to numerical values
def convert_to_numeric(value):
    try:
        return pd.to_numeric(value, errors='coerce')
    except:
        return float('nan')


merged_df['ORIGIN_HourlyVisibility'] = merged_df['ORIGIN_HourlyVisibility'].apply(convert_to_numeric)
merged_df['ORIGIN_HourlyWetBulbTemperature'] = merged_df['ORIGIN_HourlyWetBulbTemperature'].apply(convert_to_numeric)
merged_df['ORIGIN_HourlyWindDirection'] = merged_df['ORIGIN_HourlyWindDirection'].apply(convert_to_numeric)
merged_df['ORIGIN_HourlyRelativeHumidity'] = merged_df['ORIGIN_HourlyRelativeHumidity'].apply(convert_to_numeric)
merged_df['ORIGIN_HourlyPrecipitation'] = merged_df['ORIGIN_HourlyPrecipitation'].apply(convert_to_numeric)
merged_df['ORIGIN_HourlyDryBulbTemperature'] = merged_df['ORIGIN_HourlyDryBulbTemperature'].apply(convert_to_numeric)
merged_df['ORIGIN_HourlyDewPointTemperature'] = merged_df['ORIGIN_HourlyDewPointTemperature'].apply(convert_to_numeric)

# merged_df['ORIGIN_HourlyVisibility'] = merged_df['ORIGIN_HourlyVisibility'].applymap(convert_to_numeric)

merged_df['DEST_HourlyVisibility'] = merged_df['DEST_HourlyVisibility'].apply(convert_to_numeric)
merged_df['DEST_HourlyWetBulbTemperature'] = merged_df['DEST_HourlyWetBulbTemperature'].apply(convert_to_numeric)
merged_df['DEST_HourlyWindDirection'] = merged_df['DEST_HourlyWindDirection'].apply(convert_to_numeric)
merged_df['DEST_HourlyRelativeHumidity'] = merged_df['DEST_HourlyRelativeHumidity'].apply(convert_to_numeric)
merged_df['DEST_HourlyPrecipitation'] = merged_df['DEST_HourlyPrecipitation'].apply(convert_to_numeric)
merged_df['DEST_HourlyDryBulbTemperature'] = merged_df['DEST_HourlyDryBulbTemperature'].apply(convert_to_numeric)
merged_df['DEST_HourlyDewPointTemperature'] = merged_df['DEST_HourlyDewPointTemperature'].apply(convert_to_numeric)

merged_df.dropna(inplace=True)

merged_df.to_csv('merged_dataset.csv')
merged_df.columns
merged_df.head(1)
