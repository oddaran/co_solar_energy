# import libraries for data
import matplotlib.pyplot as plt
import pandas as pd

from pandas import read_csv
from datetime import datetime
from IPython.display import Image, HTML
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

import arcgis
from arcgis.gis import GIS
from arcgis.learn import FullyConnectedNetwork, MLModel, prepare_tabulardata

# connect to ArcGIS
gis = GIS(profile="DARAN_THACH_LearnArcGIS")

# Training Set
# Access Solar Dataset feature layer for Training, without the Southland Solar Plant which is hold out for validation
calgary_no_southland_solar = gis.content.search('calgary_no_southland_solar owner:api_data_owner', 'feature layer')[0]
calgary_no_southland_solar

# Access the layer from the feature layer
calgary_no_southland_solar_layer = calgary_no_southland_solar.layers[0]

# Plot location of the 10 Solar sites in Calgary to be used for training
m1 = gis.map('calgary', zoomlevel=10)
m1.add_layer(calgary_no_southland_solar_layer)
m1

# Visualize the dataframe using MODIS/Daymet observations
calgary_no_southland_solar_layer_sdf = calgary_no_southland_solar_layer.query().sdf
calgary_no_southland_solar_layer_sdf=calgary_no_southland_solar_layer_sdf[['FID','date','ID','solar_plan','altitude_m',
                                                                           'latitude','longitude','wind_speed','dayl__s_',
                                                                           'prcp__mm_d','srad__W_m_','swe__kg_m_', 'tmax__deg',
                                                                           'tmin__deg','vp__Pa_','kWh_filled','capacity_f',
                                                                           'SHAPE']]
calgary_no_southland_solar_layer_sdf.head()

# Plot & visualize the variables from training set for one solar station - Hillhurst Sunnyside Community Association
hillhurst_solar = calgary_no_southland_solar_layer_sdf[calgary_no_southland_solar_layer_sdf['solar_plan']=='Hillhurst Sunnyside Community Association'].copy()
hillhurst_datetime = hillhurst_solar.set_index(hillhurst_solar['date'])
hillhurst_datetime = hillhurst_datetime.sort_index()
for i in range(7,hillhurst_datetime.shape[1]-1):
        plt.figure(figsize=(20,3))
        plt.title(hillhurst_datetime.columns[i])
        plt.plot(hillhurst_datetime[hillhurst_datetime.columns[i]])
        plt.show()

# checking the correlation matrix between the predictors and the dependent variable of capacity_factor
corr_test = calgary_no_southland_solar_layer_sdf.drop(['FID','date','ID','latitude','longitude','solar_plan','kWh_filled'], axis=1)
corr = corr_test.corr()
corr.style.background_gradient(cmap='Greens').set_precision(2)

# Validation set
# Access the Southland Solar Plant Dataset feature layer for validation
southland_solar = gis.content.search('southland_solar owner:api_data_owner', 'feature layer')[0]
southland_solar

# Access the layer from the feature layer
southland_solar_layer = southland_solar.layers[0]

#  Plot location of the Southland Solar site in Calgary to be used for validation
m1 = gis.map('calgary', zoomlevel=10)
m1.add_layer(southland_solar_layer)
m1

# Visualize the Southland dataframe here
southland_solar_layer_sdf = southland_solar_layer.query().sdf
southland_solar_layer_sdf.head(2)

# Check the total number of samples
southland_solar_layer_sdf.shape

# Model build with ArcGIS featuring FullyConnectedNetwork (ANN), MLModel (ML)
# 1- Fullyconnectednetwork (ANN)
# This list is created naming all fields containing the predictors from the input feature layer
X = ['altitude_m', 'wind_speed', 'dayl__s_', 'prcp__mm_d','srad__W_m_','swe__kg_m_','tmax__deg','tmin__deg','vp__Pa_']

# Import the libraries from arcgis.learn for data preprocessing
from arcgis.learn import prepare_tabulardata

# Preprocess data using prepare data method to impute missing values, normalize and train-test split
data = prepare_tabulardata(calgary_no_southland_solar_layer,
                           'capacity_f',
                           explanatory_variables=X)

# Visualizing the prepared data
data.show_batch()

# Importing the model from arcgis.learn
from arcgis.learn import FullyConnectedNetwork

# Initialize the model with the data where the weights are randomly allocated
fcn = FullyConnectedNetwork(data)

# searching for an optimal learning rate using the lr_find for passing it to the final model fitting
fcn.lr_find()

# Train the Fullyconnectednetwork model
# Train model for 100 epochs
fcn.fit(100,0.0005754399373371565)

# Plot the train vs valid losses to check quality of trained model
fcn.plot_losses()

# Finally, show predicted values of model for the test set
fcn.show_results()

# Return the r-square from the model.score method to determine well training
r_Square_fcn_test = fcn.score()
print('r_Square_fcn_test: ', round(r_Square_fcn_test,5))

# Use Fullyconnectednetwork model to predict values
# Use the predict function
southland_solar_layer_predicted = fcn.predict(southland_solar_layer, output_layer_name='prediction_layer')

# Print the predicted layer
southland_solar_layer_predicted

# Access, visualize the dataframe from the predicted layer
test_pred_layer = southland_solar_layer_predicted.layers[0]
test_pred_layer_sdf = test_pred_layer.query().sdf
test_pred_layer_sdf.head()

test_pred_layer_sdf.shape

# Convert the capacity factor to values in KWh -peak capacity of Southland Leisure Centre is 153KWp
test_pred_datetime = test_pred_layer_sdf[['field1','capacity_f','prediction']].copy()
test_pred_datetime = test_pred_datetime.rename(columns={'field1':'date'})
test_pred_datetime['date'] = pd.to_datetime(test_pred_datetime['date'])
test_pred_datetime = test_pred_datetime.set_index(test_pred_datetime['date'])
test_pred_datetime['Actual_generation(KWh)'] = test_pred_datetime['capacity_f']*24*153
test_pred_datetime['predicted_generation(KWh)'] = test_pred_datetime['prediction']*24*153
test_pred_datetime = test_pred_datetime.drop(['date','capacity_f','prediction'], axis=1).sort_index()
test_pred_datetime

# Estimate model metrics of r-square, RMSE and MSE for the actual and predicted values for daily energy generation
from sklearn.metrics import r2_score
r2_test = r2_score(test_pred_datetime['Actual_generation(KWh)'],test_pred_datetime['predicted_generation(KWh)'])
print('R-Square: ', round(r2_test, 2))

# Compare between the actual sum of the total energy generated to total predicted values
actual = (test_pred_datetime['Actual_generation(KWh)'].sum()/4/1000).round(2)
predicted = (test_pred_datetime['predicted_generation(KWh)'].sum()/4/1000).round(2)
print('Actual annual Solar Energy Generated by Southland Solar Station: {} MWh'.format(actual))
print('Predicted annual Solar Energy Generated by Southland Solar Stations: {} MWh'.format(predicted))

# Plot actual vs predicted values of ANN
plt.figure(figsize=(30,6))
plt.plot(test_pred_datetime['Actual_generation(KWh)'],  linewidth=1, label= 'Actual')
plt.plot(test_pred_datetime['predicted_generation(KWh)'], linewidth=1, label= 'Predicted')
plt.ylabel('Solar Energy in KWh', fontsize=14)
plt.legend(fontsize=14,loc='upper right')
plt.title('Actual Vs Predicted Solar Energy Generated by Southland Solar-FulyConnectedNetwork Model', fontsize=14)
plt.grid()
plt.show()

# 2- MLModel (ML)
# MLM from the skikit-learn library from preprocessed data from ArcGIS.learn
# Import the libraries from arcgis.learn for data preprocessing
from sklearn.preprocessing import MinMaxScaler

# Scale feature data using MinMaxScaler()
X = ['altitude_m', 'wind_speed', 'dayl__s_', 'prcp__mm_d','srad__W_m_','swe__kg_m_','tmax__deg','tmin__deg','vp__Pa_']
preprocessors =  [('altitude_m', 'wind_speed', 'dayl__s_', 'prcp__mm_d','srad__W_m_','swe__kg_m_','tmax__deg',
                   'tmin__deg','vp__Pa_', MinMaxScaler())]

# Import prepare tabular data library
from arcgis.learn import prepare_tabulardata

# Preprocess data using prepare data method for MLModel
data = prepare_tabulardata(calgary_no_southland_solar_layer,
                           'capacity_f',
                           explanatory_variables=X,
                           preprocessors=preprocessors)

# Ensure data is being trained
data.show_batch()

# ML model initialization
# Data from tabular prepared data ready for machine learning using gradient boosting
from arcgis.learn import MLModel

# defining the model along with the parameters
model = MLModel(data, 'sklearn.ensemble.GradientBoostingRegressor', n_estimators=100, random_state=43)

model.fit()
model.show_results()

# Estimate r-squared value using model.score() from the tabular learner
print('r_square_test_rf: ', round(model.score(), 5))

# Explain the relative importance of each variable with feature_importance
import seaborn as sns

feature_imp_RF = model.feature_importances_
rel_feature_imp = 100 * (feature_imp_RF / max(feature_imp_RF))
rel_feature_imp = pd.DataFrame({'features':list(X), 'rel_importance':rel_feature_imp })

rel_feature_imp = rel_feature_imp.sort_values('rel_importance', ascending=False)

plt.figure(figsize=[15,4])
plt.yticks(fontsize=10)
ax = sns.barplot(x="rel_importance", y="features", data=rel_feature_imp, palette="BrBG")

plt.xlabel("Relative Importance", fontsize=10)
plt.ylabel("Features", fontsize=10)
plt.show()

# Predict solar generation using MLModel after gradient boosting
southland_solar_layer_predicted_rf = model.predict(southland_solar_layer, output_layer_name='prediction_layer_rf')

# Print predicted layer
southland_solar_layer_predicted_rf

# Access & visualize the dataframe from the predicted layer
valid_pred_layer = southland_solar_layer_predicted_rf.layers[0]
valid_pred_layer_sdf = valid_pred_layer.query().sdf
valid_pred_layer_sdf.head()

# Convert the capacity factor to values in KWh -peak capacity of Southland Leisure Centre is 153KWp
valid_pred_datetime = valid_pred_layer_sdf[['field1','capacity_f','prediction']].copy()
valid_pred_datetime = valid_pred_datetime.rename(columns={'field1':'date'})
valid_pred_datetime['date'] = pd.to_datetime(valid_pred_datetime['date'])
valid_pred_datetime = valid_pred_datetime.set_index(valid_pred_datetime['date'])
valid_pred_datetime['Actual_generation(KWh)'] = valid_pred_datetime['capacity_f']*24*153
valid_pred_datetime['predicted_generation(KWh)'] = valid_pred_datetime['prediction']*24*153
valid_pred_datetime = valid_pred_datetime.drop(['date','capacity_f','prediction'], axis=1)
valid_pred_datetime = valid_pred_datetime.sort_index()
valid_pred_datetime.head()

# Estimate model metrics of r-square, RMSE and MSE for the actual and predicted values for daily energy generation
from sklearn.metrics import r2_score
r2_test = r2_score(valid_pred_datetime['Actual_generation(KWh)'],valid_pred_datetime['predicted_generation(KWh)'])
print('R-Square: ', round(r2_test, 2))

# Comparison between the actual sum of the total energy generated to the total predicted values by the MLModel
actual = (valid_pred_datetime['Actual_generation(KWh)'].sum()/4/1000).round(2)
predicted = (valid_pred_datetime['predicted_generation(KWh)'].sum()/4/1000).round(2)
print('Actual annual Solar Energy Generated by Southland Solar Station: {} MWh'.format(actual))
print('Predicted annual Solar Energy Generated by Southland Solar Stations: {} MWh'.format(predicted))

# Plot actual vs predicted values of MLModel
plt.figure(figsize=(30,6))
plt.plot(valid_pred_datetime['Actual_generation(KWh)'],  linewidth=1, label= 'Actual')
plt.plot(valid_pred_datetime['predicted_generation(KWh)'], linewidth=1, label= 'Predicted')
plt.ylabel('Solar Energy in KWh', fontsize=14)
plt.legend(fontsize=14,loc='upper right')
plt.title('Actual Vs Predicted Solar Energy Generated by Southland Solar-FulyConnectedNetwork Model', fontsize=14)
plt.grid()
plt.show()
