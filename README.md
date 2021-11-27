# co_solar_energy
Predicting renewable solar generation in Colorado PV plants with ArcGIS data and solar photovoltaic plant capacity.

Introduction

Energy data analytics helps utility firms tackle major environmental, social and governance criteria, and to make top-down operational and financial decisions.  We can capitalize on the business value of solar production providers through analytics that use historical weather variables and solar production. An application for such optimization would be a forecast of renewable energy using solar photovoltaic (PV) sources, fed by mapping software application program interface (API) and machine learning models with Python. Our goal is to forecast how much energy (in kWh) will be generated from weather variables.  We obtain our datasets from existing Colorado solar PV plants.  Using the forecast of energy generation will allow a firm to make better cost-benefit analyses and long-term investment.

Business understanding

Across the USA, energy and utility companies are under pressure to generate clean energy provisioning at an affordable cost/MWh. Here in Colorado, government subsidies and tax breaks have incentivized utilities to divert from traditional fossil fuels and to utilize more renewable sources of power. With improved energy capture from solar technologies, the industry presents itself with many opportunities to meet renewable energy goals (Orsted, 2020).  Under certain circumstances, selected renewable technologies (solar, wind, and nuclear) are cost-competitive to conventional coal and gas combined sources (Lazard, 2020).  In fact, utility-scale solar photovoltaic (PV) levelized cost of energy is $31/MWh compared to $41/MWh for coal-derived, $26/MWh for wind and $29/MWh for nuclear power (Lazard, 2020).  A recent forecast (SEIA, 2020) predicts 324 GW of solar PV capacity installed over the next 10 years, a threefold increase of capacity from the previous 10 years (figure 1).

Defining data problem and understanding

The future model will help to maximize allocation of resources to accommodate solar energy production to offset net carbon emissions.  Our dataset includes existing commercial and utility solar plants in Colorado.  There are two necessary sets of data—solar plant capacity and daily weather variables.  As the study involves location and weather variables, we depend on the ArcGIS online maps (ESRI, 2021), US Energy Information Administration power plant data (EIA, 2021), and Daymet surface weather data (Thornton et al, 2021). Accessible variables include dates, kilowatt-hours (kWh), and weather inputs.  The model can visualize the distribution of one single solar PV plant and the daily generated kWh output. Plant input and weather inputs are (Thornton et al, 2021): 
•	Generation capacity factor (capacity_f)
•	Day length (dayl_s_)
•	Shortwave radiation (srad_W_m_)
•	Max air temperature (tmax_deg)
•	Min air temperature (tmin_deg)
•	Precipitation (prcp__mm_d)
•	Snow water equivalent (swe_kg_m_)
•	Water vapor pressure (vp_Pa_)
•	Wind speed (wind_speed)

Predictive analytics solution and data preparation

There are currently 98 active commercial and utility solar PV plants in Colorado operating 1756 MW of cumulative capacity (SEIA, 2020).  With known power capacities and surface weather data from tabular data, we predict daily energy production using two models (Matthews, 2019):
 •	FullyConnectedNetwork – an artificial neural network, available from the arcgis.learn module in the ArcGIS API for Python.  The model feeds feature layer or raster data into the network for classification.
 •	MLModel – a regression class from scikit-learn library from the arcgis.learn module in the API for Python. We incorporate this model from the library by passing the name of the algorithm and its relevant parameters as keyword arguments.
The dataset covers a period from September 2016 to December 2020.  The measure variables were preprocessed to obtain the main dataset used for this sample.  Two feature layers were then created from this dataset, one for training and the other for validating.  The prepared data will normalize and train-test split the data.  Once achieved, the data frame will move on for neural network and linear regression prediction modeling.

Modeling 

When the training and validation datasets are processed, analyzed, we will run the forecast with a few predictive models, namely the logistic regression method.  As we are concerned with variables that affect energy generation, criteria are set that forecasts energy for optimal resource allocation.
To preprocess our data, we consider that all the variables are continuous.  Python should pass the true value for any categorical variables.  We make a list containing the feature data from our weather inputs and plant generating capacity.  Preprocessing of the data is carried out by the prepare_tabulardata method from the arcgis.learn module.  This output gives a tabulardataobject to input into the neural network and regression models.  The explanatory variables will create a data frame for the model with these parameters:

•	input_features: the spatial dataframe containing the primary dataset
•	variable_predict: fields containing predictor variable from the input dataframe
•	explanatory_variables: list of the fields as tuples of explanatory variables above

Model build

Our ArcGIS-enabled model generates a time series which will provide seasonality forecasts, with correlation plots to look at variable correlation.  Building such models gives a sense of what accuracy can be expected without applying any additional effort, e.g., a baseline.  In this case, we have an idea which variable is the most correlated – day length and radiation.  After processing the explanatory variables from prepare_tabulardata method, we employ the neural network for training. First, the FullyConnectedNetwork is imported from arcgis.learn and initialized:

Deployment and relation to SAS Enterprise Miner

Python was selected for our solar PV production forecast due to its diverse, popular modules (matplotlib, pandas) associated with ArcGIS.  Stakeholders may access this code in open-source sites such as Github.com.  Equally as powerful are the languages Azure, R, and SAS Enterprise Miner.  For medium-scale data one can use PostgreSQL.  R language has object functionality like Python, and beautiful graphic libraries. SAS Enterprise Miner, being a closed-source platform, can perform well with its diagram drag-drop functionality and model comparisons.  However, SAS is unable to access mapping software such as ArcGIS.

Model evaluation and conclusion

This project attempted to create a predictive analytics model of energy generation (KWh) using Colorado solar PV plants, along with historical surface weather data.  Using two methods, a trained neural network and regression algorithm, we were able to run trained and validation data into these methods.  To check for overfitting, we specified several training and validation losses compared to a learning rate, to ensure the losses are gradually decreased. Then the predicted values from the trained model can be shown as test results –the output of the kWh generated by Denver area solar plants.  Then the trained neural network can predict daily solar generation for a given time range. Our model visualized relative importance of other dependent variables in a histogram chart.  The results show that day length and sunshine radiation are higher in relative importance than other surface weather metrics.  Our prediction results of both neural network and regression compared well to the historical, actual values.

Solar energy is safe, abundant and can be utilized by regional and corporate structures. This model can be applied to future forecasting studies to model energy production from sustainable renewable sources such as from photovoltaic solar.  Feng, Gong, and Zhang (2019) note that models using air temperature successfully made accurate prediction for solar radiation and are good for design of subsequent solar PV energy systems.  Moreover, as PV technology price drops, the applications of buildings and areas will offer increased benefits for renewable energy markets such as in Colorado (OEDI, 2018).  Noteworthy is that our predictive model is not limited to jurisdictions run by regional utility companies.  It may also satisfy the environmental, social, and governance criteria whereby corporate power consumers are more incentivized to use cheaper and cleaner energy.
