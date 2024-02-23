#%%[markdown]
## Modeling to predict Danceability
#%%
# Import
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from Spotify_EDA import df_danceabiltiy # Import df data frame from Spotify_EDA to use the processed data for modeling

# %%
# %%

# Selecting relevant features and the target variable
features = ['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'danceability'

# Handling missing values
spotify_data = df_danceabiltiy.dropna(subset=features + [target])

# Splitting the dataset into training and testing sets
X = spotify_data[features]
y = spotify_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}, R-squared: {r2}')
# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Selecting relevant features and the target variable
features = ['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'danceability'

# Handling missing values
spotify_data = df_danceabiltiy.dropna(subset=features + [target])

# Splitting the dataset into training and testing sets
X = spotify_data[features]
y = spotify_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Building and evaluating Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_mse, rf_r2 = evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Building and evaluating Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)
gb_mse, gb_r2 = evaluate_model(gb_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Hyperparameter tuning for Random Forest using Grid Search
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train_scaled, y_train)

# Best parameters and evaluation
best_rf_model = grid_search_rf.best_estimator_
best_rf_mse, best_rf_r2 = evaluate_model(best_rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Print the results
print("Random Forest MSE:", rf_mse, "R2:", rf_r2)
print("Gradient Boosting MSE:", gb_mse, "R2:", gb_r2)
print("Tuned Random Forest MSE:", best_rf_mse, "R2:", best_rf_r2)
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Hyperparameters to tune
param_dist_rf = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11)
}

# Randomized Search for hyperparameter tuning
random_search_rf = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist_rf, n_iter=50, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42)
random_search_rf.fit(X_train_scaled, y_train)

# Best parameters and evaluation
best_rf_model = random_search_rf.best_estimator_
best_rf_mse_ex, best_rf_r2_ex = evaluate_model(best_rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

print("Tuned Random Forest (Extended) MSE:", best_rf_mse_ex, "R2:", best_rf_r2_ex)
print("Best Random Forest (Extended) Parameters:", random_search_rf.best_params_)

import xgboost as xgb

# Defining the XGBoost regressor
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)

# Hyperparameter grid
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.7, 0.8],
    'subsample': [0.7, 0.8]
}

# Grid Search for hyperparameter tuning
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train_scaled, y_train)

# Best parameters and evaluation
best_xgb_model = grid_search_xgb.best_estimator_
best_xgb_mse, best_xgb_r2 = evaluate_model(best_xgb_model, X_train_scaled, y_train, X_test_scaled, y_test)

print("XGBoost MSE:", best_xgb_mse, "R2:", best_xgb_r2)
print("Best XGBoost Parameters:", grid_search_xgb.best_params_)

# %%
# Extracted MSE and R2 values from the user's code
mse_values = [mse, rf_mse, gb_mse, best_rf_mse, best_rf_mse_ex,best_xgb_mse]
r2_values = [r2, rf_r2, gb_r2, best_rf_r2, best_rf_r2_ex, best_xgb_r2]

# Model names
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting',
          'Tuned Random Forest', 'Tuned RF (Extended)', 'XGBoost']

# Creating bar plots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# MSE Bar Chart
ax[0].bar(models, mse_values, color='skyblue')
ax[0].set_title('Model MSE Comparison')
ax[0].set_ylabel('Mean Squared Error (MSE)')
ax[0].set_xticklabels(models, rotation=45, ha='right')

# R2 Bar Chart
ax[1].bar(models, r2_values, color='lightgreen')
ax[1].set_title('Model R-squared Comparison')
ax[1].set_ylabel('R-squared Value')
ax[1].set_xticklabels(models, rotation=45, ha='right')

plt.tight_layout()
plt.show()
# %%[markdown]
## 1. Linear Regression
#
#MSE: 0.0203
#
#R-squared: 0.3262
#
#Analysis:
#The Linear Regression model, with an MSE of 0.0203, indicates a moderate level of prediction error.
#An R-squared value of 0.3262 suggests that approximately 32.62% of the variance in danceability can be explained by the model.
#Interpretation: This model provides a basic level of prediction capability but isn't highly accurate in capturing the complexities of the data.
#
## 2. Random Forest Regressor
#
#MSE: 0.0087
#
#R-squared: 0.7115
#
#Analysis:
#The Random Forest Regressor significantly improves the prediction with a lower MSE of 0.0087, indicating more accurate predictions.
# A higher R-squared value of 0.7115 shows that it can explain about 71.15% of the variance.
# Interpretation: This model, with its ensemble approach, captures more nuances in the data compared to the Linear Regression model.
#
## 3. Gradient Boosting Regressor
#
# MSE: 0.0135
#
# R-squared: 0.5521
#
# Analysis:
# MSE of 0.0135 is an improvement over Linear Regression but not as good as Random Forest.
# R-squared value of 0.5521 indicates it explains about 55.21% of the variance.
# Interpretation: Gradient Boosting provides a balance between the simplicity of Linear Regression and the complexity of Random Forest.
#
## 4. Tuned Random Forest Regressor
#
# MSE: 0.0086
#
# R-squared: 0.7136
#
# Analysis:
# The MSE improves slightly to 0.0086, and R-squared increases to 0.7136 in the tuned model.
# Interpretation: Fine-tuning the Random Forest model has led to a marginal but notable improvement, making it the most accurate model among those tested.
#
## 5. Tuned Random Forest Regressor (Extended)
#
# MSE: 0.0087
#
# R-squared: 0.7094
#
# Analysis:
# The MSE is similar to the standard Random Forest, and the R-squared is slightly lower.
# Interpretation: Extended tuning does not significantly change the performance, indicating that the basic tuning was already quite effective.
#
## 6. XGBoost Regressor
#
# MSE: 0.0100
#
# R-squared: 0.6681
#
# Analysis:
# The MSE is higher than the Tuned Random Forest, indicating slightly less accurate predictions.
# An R-squared of 0.6681 is respectable but not the highest among the models.
# Interpretation: XGBoost performs well, but in this case, it's slightly outperformed by the Tuned Random Forest model.
#
## Overall Conclusion
# The Tuned Random Forest Regressor stands out as the most effective model for this task, achieving the lowest MSE and highest R-squared value. It indicates a strong balance between accuracy and the ability to explain the variance in danceability.
# Each model has its strengths and weaknesses, and the choice of model can depend on the specific requirements of the task at hand, such as the need for interpretability (Linear Regression) vs. predictive power (Random Forest, XGBoost).
# %%
