import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv('airport_dataset.csv')
d = d.drop('Unnamed: 0',axis = 1)

plt.hist(d['DEP_DELAY'])
plt.show()

# IMPORTING ENCODER
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# Create a label encoder
le = LabelEncoder()

# Fit the label encoder to the column
le.fit(d['OP_UNIQUE_CARRIER'])

# Transform the column
d['OP_UNIQUE_CARRIER'] = le.transform(d['OP_UNIQUE_CARRIER'])

print(d)

# convert data into numpy array and create 2 datasets for arrival and departure delays

df_delays = d.iloc[:, 0:7]
# Get the column names
cols = list(df_delays.columns)

# Swap the last and 2nd last columns
cols[-1], cols[-2] = cols[-2], cols[-1]

# Reindex the DataFrame
df_delays = df_delays.reindex(columns=cols)



df_arrivals = d.drop('DEP_DELAY', axis=1)

# 80-20 split for departure delays
from sklearn import model_selection
dep_data = df_delays.to_numpy()
x_dep = dep_data[:, :-1]
y_dep = dep_data[:, -1]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_dep, y_dep, train_size= 0.8, random_state= 42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
# x_train_scaled[:,0].mean()

# GBDT model
gbdt = GradientBoostingRegressor(max_leaf_nodes=8, min_samples_leaf=10, learning_rate=0.05, n_estimators=1000, verbose=2)

# Fit the GradientBoostingClassifier model to the data
gbdt.fit(x_train_scaled, y_train)

# Predict the values of col4
predictions = gbdt.predict(x_test_scaled)



# mean absolute error
mae = mean_absolute_error(y_test, predictions)
print('Mean absolute error:', mae)

#root mean squared error
rmse = mean_squared_error(y_test, predictions)
print('Root mean squared error:', rmse)

# R-squared error
r2 = r2_score(y_test, predictions)
print('R-squared error:', r2)


from sklearn.gaussian_process  import GaussianProcessRegressor



# Create a GaussianRegressor object
reg = GaussianProcessRegressor()

# Fit the GaussianRegressor model to the data
reg.fit(x_train_scaled, y_train)

# Predict the values of col4
predictions = reg.predict(x_test_scaled)