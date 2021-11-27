# Import libraries for data
import matplotlib.pyplot as plt
import pandas as pd

import arcgis
from arcgis.gis import GIS
from arcgis.learn import FullyConnectedNetwork, MLModel, prepare_tabulardata

# Connect to ArcGIS
gis = GIS(profile="DARAN_THACH_LearnArcGIS")

# Training Set
# Access Solar Dataset feature layer for Training, without the Southland Solar Plant which is hold out for validation
co_solar_no_denver = gis.content.search('co_solar_no_denver owner:api_data_owner', 'feature layer')[0]
co_solar_no_denver

# Access the layer from the feature layer
co_solar_no_denver_layer = co_solar_no_denver.layers[0]

# Plot location of the various Colorado (excl Denver Metro) solar plants for training
m1 = gis.map('colorado', zoomlevel=7)
m1.add_layer(co_solar_no_denver_layer)
m1

# Visualize the dataframe using MODIS/Daymet observations
co_solar_no_denver_layer_sdf = co_solar_no_denver_layer.query().sdf
co_solar_no_denver_layer_sdf=co_solar_no_denver_layer_sdf[['FID','date','ID','solar_plan','altitude_m',
                                                                           'latitude','longitude','wind_speed','dayl__s_',
                                                                           'prcp__mm_d','srad__W_m_','swe__kg_m_', 'tmax__deg',
                                                                           'tmin__deg','vp__Pa_','kWh_filled','capacity_f',
                                                                           'SHAPE']]
co_solar_no_denver_layer_sdf.head()

# Plot & visualize the variables from training set for one solar station - SR Jenkins Ft Lupton
srjenkins_solar = co_solar_no_denver_layer_sdf[co_solar_no_denver_layer_sdf['solar_plan']=='SR Jenkins Ft Lupton'].copy()
srjenkins_datetime = srjenkins_solar.set_index(srjenkins_solar['date'])
srjenkins_datetime = srjenkins_datetime.sort_index()
for i in range(7,srjenkins_datetime.shape[1]-1):
        plt.figure(figsize=(20,3))
        plt.title(srjenkins_datetime.columns[i])
        plt.plot(srjenkins_datetime[srjenkins_datetime.columns[i]])
        plt.show()

# checking the correlation matrix between the predictors and the dependent variable of capacity_factor
corr_test = co_solar_no_denver_layer_sdf.drop(['FID','date','ID','latitude','longitude','solar_plan','kWh_filled'], axis=1)
corr = corr_test.corr()
corr.style.background_gradient(cmap='Greens').set_precision(2)

# Validation set
# Access the Denver-metro only solar plant dataset feature layer for validation
co_solar_denver = gis.content.search('co_solar_denver owner:api_data_owner', 'feature layer')[0]
co_solar_denver

# Access the layer from the feature layer
co_solar_denver_layer = co_solar_denver.layers[0]

#  Plot location of the Denver-metro solar plants used for validation
m1 = gis.map('colorado', zoomlevel=7)
m1.add_layer(co_solar_denver_layer)
m1

# Visualize the Denver-metro dataframe here
co_solar_denver_layer_sdf = co_solar_denver_layer.query().sdf
co_solar_denver_layer_sdf.head(2)

# Check the total number of samples
co_solar_denver_layer_sdf.shape

# Model build with ArcGIS featuring FullyConnectedNetwork (ANN), MLModel (ML)
# 1- Fullyconnectednetwork (ANN)
# This list is created naming all fields containing the predictors from the input feature layer
X = ['altitude_m', 'wind_speed', 'dayl__s_', 'prcp__mm_d','srad__W_m_','swe__kg_m_','tmax__deg','tmin__deg','vp__Pa_']

# Import the libraries from arcgis.learn for data preprocessing
from arcgis.learn import prepare_tabulardata

# Preprocess data using prepare data method to impute missing values, normalize and train-test split
data = prepare_tabulardata(co_solar_no_denver_layer,
                           'capacity_f',
                           explanatory_variables=X)

# Visualizing the prepared data
data.show_batch()

# Importing the model from arcgis.learn
from arcgis.learn import FullyConnectedNetwork

# Initialize the model with the data where the weights are randomly allocated
fcn = FullyConnectedNetwork(data)

# Search for an optimal learning rate with lr_find for passing it to the final model fitting
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
co_solar_denver_layer_predicted = fcn.predict(co_solar_denver_layer, output_layer_name='prediction_layer')

# Print the predicted layer
co_solar_denver_layer_predicted

# Access, visualize the dataframe from the predicted layer
test_pred_layer = co_solar_denver_layer_predicted.layers[0]
test_pred_layer_sdf = test_pred_layer.query().sdf
test_pred_layer_sdf.head()

test_pred_layer_sdf.shape

# Convert the capacity factor to values in KWh -peak capacity of SR Jenkins Ft Lupton is 13MW
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
print('Actual annual Solar Energy Generated by Denver Metro Solar Stations: {} MWh'.format(actual))
print('Predicted annual Solar Energy Generated by Denver Metro Solar Stations: {} MWh'.format(predicted))

# Plot actual vs predicted values of ANN
plt.figure(figsize=(30,6))
plt.plot(test_pred_datetime['Actual_generation(KWh)'],  linewidth=1, label= 'Actual')
plt.plot(test_pred_datetime['predicted_generation(KWh)'], linewidth=1, label= 'Predicted')
plt.ylabel('Solar Energy in KWh', fontsize=14)
plt.legend(fontsize=14,loc='upper right')
plt.title('Actual vs Predicted Solar Energy Generated by Denver Metro-Artificial Neural Network Model', fontsize=14)
plt.grid()
plt.show()

# 2- MLModel (ML)
# ML model from the skikit-learn library from preprocessed data from ArcGIS.learn
# Import the libraries from arcgis.learn for data preprocessing
from sklearn.preprocessing import MinMaxScaler

# Scale feature data using MinMaxScaler()
X = ['altitude_m', 'wind_speed', 'dayl__s_', 'prcp__mm_d','srad__W_m_','swe__kg_m_','tmax__deg','tmin__deg','vp__Pa_']
preprocessors =  [('altitude_m', 'wind_speed', 'dayl__s_', 'prcp__mm_d','srad__W_m_','swe__kg_m_','tmax__deg',
                   'tmin__deg','vp__Pa_', MinMaxScaler())]

# Import prepare tabular data library
from arcgis.learn import prepare_tabulardata

# Preprocess data using prepare data method for MLModel
data = prepare_tabulardata(co_solar_no_denver_layer,
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
co_solar_denver_layer_predicted_rf = model.predict(co_solar_denver_layer, output_layer_name='prediction_layer_rf')

# Print predicted layer
co_solar_denver_layer_predicted_rf

# Access & visualize the dataframe from the predicted layer
valid_pred_layer = co_solar_denver_layer_predicted_rf.layers[0]
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
print('Actual annual Solar Energy Generated by Denver Metro Solar Station: {} MWh'.format(actual))
print('Predicted annual Solar Energy Generated by Denver Metro Solar Stations: {} MWh'.format(predicted))

# Plot actual vs predicted values of MLModel
plt.figure(figsize=(30,6))
plt.plot(valid_pred_datetime['Actual_generation(KWh)'],  linewidth=1, label= 'Actual')
plt.plot(valid_pred_datetime['predicted_generation(KWh)'], linewidth=1, label= 'Predicted')
plt.ylabel('Solar Energy in KWh', fontsize=14)
plt.legend(fontsize=14,loc='upper right')
plt.title('Actual vs Predicted Solar Energy Generated by Denver Metro-Machine Learning Model', fontsize=14)
plt.grid()
plt.show()
