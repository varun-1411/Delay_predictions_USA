import pandas as pd
import torch
import gpytorch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from math import floor

from Airdelays_preprocessing.main import five_largest_airports

tail_numbers = five_largest_airports['TAIL_NUM'].unique()
tail_number_to_index = {tail: index for index, tail in enumerate(tail_numbers)}
five_largest_airports['TAIL_NUM_INDEX'] = five_largest_airports['TAIL_NUM'].map(tail_number_to_index)

flight_numbers = five_largest_airports['OP_CARRIER_FL_NUM'].unique()
flight_number_to_index = {flight: index for index, flight in enumerate(flight_numbers)}
five_largest_airports['FLIGHT_NUM_INDEX'] = five_largest_airports['OP_CARRIER_FL_NUM'].map(flight_number_to_index)

five_largest_airports['CARRIER_DELAY'] = five_largest_airports['CARRIER_DELAY'].fillna(0)
five_largest_airports['WEATHER_DELAY'] = five_largest_airports['WEATHER_DELAY'].fillna(0)
five_largest_airports['NAS_DELAY'] = five_largest_airports['NAS_DELAY'].fillna(0)
five_largest_airports['SECURITY_DELAY'] = five_largest_airports['SECURITY_DELAY'].fillna(0)
five_largest_airports['LATE_AIRCRAFT_DELAY'] = five_largest_airports['LATE_AIRCRAFT_DELAY'].fillna(0)


# five_largest_airports['Number of Flights_dep'].dropna(inplace=True)

def convert_time_to_min(time_str):
    time_str = str(time_str).zfill(4)

    hours = int(time_str[:2])
    mins = int(time_str[2:])
    total_mins = hours * 60 + mins
    return total_mins


five_largest_airports['CRS_DEP_TIME'] = five_largest_airports['CRS_DEP_TIME'].apply(convert_time_to_min)
five_largest_airports['CRS_ARR_TIME'] = five_largest_airports['CRS_ARR_TIME'].apply(convert_time_to_min)

five_largest_airports.drop(columns=['FL_DATE', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'timestamp'])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Suppose your data is in a DataFrame called df

# Create a label encoder instance
# onehot_encoder = OneHotEncoder(sparse=False)
five_largest_airports = pd.read_csv('airports.csv')
# Fit and transform the 'origin' and 'destination' columns
origin_encoder = OneHotEncoder(sparse=False)
dest_encoder = OneHotEncoder(sparse=False)

# Read the CSV file
five_largest_airports = pd.read_csv('airports.csv')

# Fit and transform the 'origin' and 'destination' columns with separate encoders
origin_encoded_data = origin_encoder.fit_transform(five_largest_airports[['ORIGIN']])
dest_encoded_data = dest_encoder.fit_transform(five_largest_airports[['DEST']])

# Get the unique category names for 'ORIGIN' and 'DEST'
origin_categories = origin_encoder.categories_[0]
dest_categories = dest_encoder.categories_[0]

# Create new columns in the DataFrame for the encoded values with different names
origin_encoded_columns = ['origin_' + category for category in origin_categories]
dest_encoded_columns = ['dest_' + category for category in dest_categories]

# Assign the encoded data to the new columns
five_largest_airports[origin_encoded_columns] = origin_encoded_data
five_largest_airports[dest_encoded_columns] = dest_encoded_data

# convert all delays less than zero to zero
five_largest_airports['DEP_DELAY'] = five_largest_airports['DEP_DELAY'].clip(lower=0)

# remove all delays above 300 min
five_largest_airports = five_largest_airports[five_largest_airports['DEP_DELAY'] < 300]

train_data = five_largest_airports[five_largest_airports['DAY_OF_MONTH'] < 30]
test_data = five_largest_airports[five_largest_airports['DAY_OF_MONTH'] >= 30]

train_features = train_data[[ 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME',
                             'origin_ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT, GA US',
                             'origin_CHICAGO OHARE INTERNATIONAL AIRPORT, IL US',
                             'origin_DAL FTW WSCMO AIRPORT, TX US',
                             'origin_DENVER CENTENNIAL AIRPORT, CO US',
                             'dest_ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT, GA US',
                             'dest_CHICAGO OHARE INTERNATIONAL AIRPORT, IL US',
                             'dest_DAL FTW WSCMO AIRPORT, TX US',
                             'dest_DENVER CENTENNIAL AIRPORT, CO US','Number of Flights_arr',
                              'Number of Flights_dep']].values
test_features = test_data[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME',
                             'origin_ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT, GA US',
                             'origin_CHICAGO OHARE INTERNATIONAL AIRPORT, IL US',
                             'origin_DAL FTW WSCMO AIRPORT, TX US',
                             'origin_DENVER CENTENNIAL AIRPORT, CO US',
                             'dest_ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT, GA US',
                             'dest_CHICAGO OHARE INTERNATIONAL AIRPORT, IL US',
                             'dest_DAL FTW WSCMO AIRPORT, TX US',
                             'dest_DENVER CENTENNIAL AIRPORT, CO US','Number of Flights_arr',
                           'Number of Flights_dep']].values

# train_features = train_data[['DAY_OF_WEEK','CRS_DEP_TIME']].values
# test_features = test_data[['DAY_OF_WEEK','CRS_DEP_TIME']].values

# Calculate mean and standard deviation for each feature in the training set
mean_train = train_features.mean(axis=0)
std_train = train_features.std(axis=0)

# Z-score normalize the training features
train_features_normalized = (train_features - mean_train) / std_train

# Z-score normalize the test features using the same mean and standard deviation
test_features_normalized = (test_features - mean_train) / std_train

# Convert normalized features back to tensors
train_x = torch.tensor(train_features_normalized, dtype=torch.float)
test_x = torch.tensor(test_features_normalized, dtype=torch.float)
train_y = torch.tensor(train_data['DEP_DELAY'].values, dtype=torch.float)
test_y = torch.tensor(test_data['DEP_DELAY'].values, dtype=torch.float)

# # Assuming you want to sample the DataFrame for train and test sets
# train_x_indices = five_largest_airports.sample(frac=0.8, random_state=1).index
# test_x_indices = five_largest_airports.index.difference(train_x_indices)
#
# train_x = train_x[train_x_indices]
# test_x = test_x[test_x_indices]

# train_x = torch.tensor((train_data[['Number of Flights_dep', 'DAY_OF_MONTH']].values), dtype=torch.float)
# train_y = torch.tensor(train_data['DEP_DELAY'].values, dtype=torch.float)
# # ['DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'Number of Flights_arr', 'Number of Flights_dep', 'FLIGHT_NUM_INDEX', "DISTANCE",'TAXI_OUT', 'origin_encoded','destination_encoded']
# test_x = torch.tensor(test_data[['Number of Flights_dep', 'DAY_OF_MONTH']].values, dtype=torch.float)
# test_y = torch.tensor(test_data['DEP_DELAY'].values, dtype=torch.float)
#
# train_x = train_x - train_x.min(0)[0]
# train_x = 2 * (train_x / train_x.max(0)[0]) - 1
# print(train_x)
# test_x = test_x - test_x.min(0)[0]
# test_x = 2 * (test_x / test_x.max(0)[0]) - 1
# print(train_x)

# train_x = five_largest_airports.sample(frac=0.8, random_state=1)
# test_x = five_largest_airports.drop(train_x.index)
#
# # Z score scale the training and test sets
# for column in train_x.columns:
#     mean = train_x[column].mean()
#     std = train_x[column].std()
#     train_x[column] = (train_x[column] - mean) / std
#     test_x[column] = (test_x[column] - mean) / std

# Print the training and test sets
print(train_x)
print(test_x)

# x = torch.tensor(five_largest_airports[
#                      ['DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'FLIGHT_NUM_INDEX', "DISTANCE", 'origin_encoded',
#                       'destination_encoded']].values, dtype=torch.float)
#
# y = torch.tensor(five_largest_airports['DEP_DELAY'].values, dtype=torch.float)
# x = x - x.min(0)[0]
# x = 2 * (x / x.max(0)[0]) - 1
#
# train_n = int(floor(0.8 * len(x)))
# train_x = x[:train_n, :]
# train_y = y[:train_n]
#
# test_x = x[train_n:, :]
# test_y = y[train_n:]

data_dim = train_x.size(-1)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=12)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.feature_extractor = feature_extractor

        # This module will scale the NN features so that they're nice values
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # projected_x = self.feature_extractor(x)
        # projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # covar_x = self.rbf_kernel_module(x) + self.white_noise_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

training_iterations = 100

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    # {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


def train():
    # iterator = 60
    iterator = tqdm.tqdm(range(training_iterations))

    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()


train()
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    preds = model(test_x)
    y_preds = likelihood(model(test_x))

print('Test MAE: {}'.format(torch.mean(torch.abs(y_preds.mean - test_y))))

print(preds)
print(y_preds.mean.numpy())
pred = torch.abs(y_preds.mean - test_y)

observed_pred = likelihood(model(test_x))
f_pred = model(test_x)
print(observed_pred)
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = preds.confidence_region()
    # Plot training data as black stars
    # ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blu
    # e line
    ax.plot(test_y.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_y.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    # ax.legend(['Observed Data', 'Mean', 'Confidence'])

plt.figure(figsize=(10, 6))
plt.scatter(test_y.numpy(), preds.mean.numpy(), color='blue', label='Predicted vs True')
# plt.show()
plt.plot([min(test_y.numpy()), max(test_y.numpy())], [min(test_y.numpy()), max(test_y.numpy())], color='red',
         linestyle='--', label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(train_y.numpy(), preds.mean.numpy())

plt.show()
