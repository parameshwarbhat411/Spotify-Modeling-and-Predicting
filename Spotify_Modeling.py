#%%[markdown]
##Modeling to predict popularity
#%%
# Import 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Spotify_EDA import df # Import df data frame from Spotify_EDA to use the processed data for modeling
from scipy.stats import skew
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Check for skewness
for col in numeric_cols:
    skewness = skew(df[col])
    print(f"Skewness of {col}: {skewness}")

# Assuming df is your DataFrame and numeric_cols contains the names of your numeric columns
for col in numeric_cols:
    skewness = skew(df[col])
    print(f"Skewness of {col}: {skewness}")

#%%[markdown]
# * duration_ms (10.81): Highly right-skewed. Consider applying a log transformation.
# * explicit (2.96): Moderately right-skewed. Investigate the distribution; if it's a boolean or binary feature, skewness might not be relevant.
# * loudness (-2.01): Moderately left-skewed. You might consider a square or cube root transformation.
# * speechiness (4.64): Highly right-skewed. A log transformation could be beneficial.
# * instrumentalness (1.74): Moderately right-skewed. Log transformation could be applied.
# * liveness (2.11): Moderately right-skewed. Log transformation is recommended.

# but we dont do transformation because we would not like to use linear model
#%%
from sklearn.model_selection import train_test_split

# Drop unnecessary columns
df = df.drop(['track_id','album_name', 'track_name'], axis=1)

# Define features and target variable
X = df.drop('popularity', axis=1)
y = df['popularity']
print(X.head(10))
print(y.head(10))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now X_train and y_train can be used to train the model, 
# and X_test and y_test to evaluate its performance
#%%
print(X.columns)
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_linear = linear_model.predict(X_test)
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_linear, squared=False))
print("Linear Regression R² Score:", r2_score(y_test, y_pred_linear))

#%%[markdown]

# * High RMSE: A higher RMSE value indicates that the model's predictions are, on average, quite far from the actual values. This suggests that the Linear Regression model might not be capturing the complexity or patterns in the data effectively.

# * Low R² Score: A score of 0.027 is very low, indicating that the model explains only about 2.7% of the variance in the target variable (popularity).
#%%
from sklearn.ensemble import RandomForestRegressor

# Initialize and train model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("Random Forest R² Score:", r2_score(y_test, y_pred_rf))

#%%[markdown]
# * Lower RMSE Compared to Linear Regression: The RMSE value is lower than that of the Linear Regression model you previously evaluated. This indicates that the Random Forest model is, on average, making predictions closer to the actual values.

# * Moderate R² Score: The R² Score is significantly higher than that of the Linear Regression model, at approximately 52.6%. This means that the Random Forest model explains about 52.6% of the variance in the target variable (popularity), which is a moderate performance.
#%%
from xgboost import XGBRegressor
# Initialize and train model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("XGBoost R² Score:", r2_score(y_test, y_pred_xgb))

#%%[markdown]
# * RMSE: The RMSE value is higher than that of the Random Forest model but still indicates a reasonable level of accuracy. A lower RMSE would be desirable as it means the model's predictions are closer to the actual values.

# * R² Score: The R² Score of approximately 42.7% means that the XGBoost model explains around 42.7% of the variance in the target variable (popularity). This is a moderate level of predictive power but lower than the Random Forest model.
#%%
from sklearn.linear_model import Lasso

# Initialize and train model
lasso_model = Lasso(random_state=42)
lasso_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lasso = lasso_model.predict(X_test)
print("Lasso Regression RMSE:", mean_squared_error(y_test, y_pred_lasso, squared=False))
print("Lasso Regression R² Score:", r2_score(y_test, y_pred_lasso))

#%%[markdown]
# * RMSE (Root Mean Square Error): This value is quite high, implying that the predictions from the Lasso Regression model are, on average, quite far from the actual values. A lower RMSE is desirable as it indicates more accurate predictions.

# * R² Score: The R² Score is very close to zero, indicating that the Lasso Regression model explains almost none of the variability of the target variable around its mean. In simple terms, it's not much better than a model that would always predict the average popularity.

# %%
from catboost import CatBoostRegressor

# Initialize and train model
cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_cat = cat_model.predict(X_test)
print("CatBoost RMSE:", mean_squared_error(y_test, y_pred_cat, squared=False))
print("CatBoost R² Score:", r2_score(y_test, y_pred_cat))

#%%[markdown]
# * RMSE (Root Mean Square Error): The RMSE value suggests that the CatBoost model's predictions are closer to the actual values compared to simpler models like Linear Regression and Lasso Regression. However, it's still not as accurate as the Random Forest model.

# * R² Score: An R² Score of approximately 0.40 indicates that around 40% of the variability in your target variable (popularity) can be explained by the CatBoost model. This is a decent score, but it shows that there is still a significant portion of the variability that the model is not capturing.
#%%
from sklearn.linear_model import Ridge

# Initialize and train model
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_ridge = ridge_model.predict(X_test)
print("Ridge Regression RMSE:", mean_squared_error(y_test, y_pred_ridge, squared=False))
print("Ridge Regression R² Score:", r2_score(y_test, y_pred_ridge))

#%%[markdown]
# * RMSE (Root Mean Square Error): The RMSE value is relatively high, which implies that the predictions from the Ridge Regression model deviate significantly from the actual values. This indicates a lower predictive accuracy compared to more complex models like Random Forest or XGBoost.

# * R² Score: An R² Score of approximately 0.027 indicates that only about 2.7% of the variability in your target variable (popularity) is explained by the Ridge Regression model. This is a very low score and suggests that the model is not capturing much of the underlying variance in the data.
#%%
from sklearn.svm import SVR

# Initialize and train model
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_svm = svr_model.predict(X_test)
print("SVR RMSE:", mean_squared_error(y_test, y_pred_svm, squared=False))
print("SVR R² Score:", r2_score(y_test, y_pred_svm))

#%%[markdown]
# * RMSE (Root Mean Square Error): The high RMSE value indicates that the predictions from the SVR model have significant errors compared to the actual values, implying lower predictive accuracy.

# * R² Score: A negative R² Score, particularly one close to zero, implies that the model performs worse than a simple horizontal line representing the mean of the target variable. This indicates that the SVR model does not adequately capture the variance in your data.

#%%
from sklearn.ensemble import VotingRegressor

# Create the sub-models
estimators = [
    ('linear', linear_model),
    ('random_forest', rf_model),
    ('xgb', xgb_model),
    ('lasso', lasso_model),
    ('catboost', cat_model),
    ('ridge', ridge_model),
    ('svr', svr_model)
]

# Create the voting regressor
voting_model = VotingRegressor(estimators, weights=[2, 3, 3, 1, 3, 1, 1])  # Weights can be adjusted

# Fit the voting regressor to the training data
voting_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_voting = voting_model.predict(X_test)
print("Voting Regressor RMSE:", mean_squared_error(y_test, y_pred_voting, squared=False))
print("Voting Regressor R² Score:", r2_score(y_test, y_pred_voting))

#%%[markdown]
# * RMSE (Root Mean Square Error): The RMSE value is a measure of the average magnitude of errors in your predictions. An RMSE of 17.46 suggests that, on average, the predictions of the Voting Regressor deviate from the actual values by this amount. It's a moderate value, indicating a reasonable level of accuracy but with room for improvement.

# * R² Score: The R² Score is a measure of how well the variations in your target variable are explained by the model. An R² Score of 0.392 suggests that the model explains approximately 39.2% of the variance in the target variable. This is a fair score but indicates that there's significant variance left unexplained by the model.
#%%
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Define base models
estimators = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgboost', XGBRegressor(n_estimators=100, random_state=42))
]

# Define final meta-learner model
final_estimator = Ridge()

# Create the Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator
)

# Fit the model
stacking_regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred_stack = stacking_regressor.predict(X_test)
print("Stacking Regressor RMSE:", mean_squared_error(y_test, y_pred_stack, squared=False))
print("Stacking Regressor R² Score:", r2_score(y_test, y_pred_stack))

#%%[markdown]
# * RMSE: The RMSE value is a measure of the average magnitude of the prediction errors. A lower RMSE value is better, and in your case, an RMSE of approximately 15.17 indicates that the predictions made by the Stacking Regressor are, on average, about 15.17 units away from the actual values. This is a comparatively good result, suggesting that the Stacking Regressor is making relatively accurate predictions.

# * R² Score: The R² Score represents the proportion of variance in the dependent variable (popularity) that is predictable from the independent variables. An R² Score of about 0.541 means that around 54.1% of the variance in your target variable is explained by the model, which is a moderate to good score. It indicates that the model has a reasonable fit to the data, though there is still some unexplained variance.
#%%

""" # Define the sub-models for VotingRegressor
estimators = [
    ('linear', linear_model),
    ('random_forest', rf_model),
    ('xgb', xgb_model),
    ('lasso', lasso_model),
    ('catboost', cat_model),
    ('ridge', ridge_model),
    ('svm', svr_model)
]

# Grid Search for Optimal Weights
from itertools import product

# Define a range of weights
weight_options = [1, 2, 3, 4, 5]

# Generate combinations of weights
weight_combinations = product(weight_options, repeat=len(estimators))

# Define a function to create a VotingRegressor with given weights
def get_voting_regressor(weights):
    return VotingRegressor(estimators, weights=weights)

# Grid search
best_score = float('inf')
best_weights = None

for weights in weight_combinations:
    model = get_voting_regressor(weights)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred, squared=False)
    if score < best_score:
        best_score = score
        best_weights = weights

print("Best Weights:", best_weights)
print("Best Score:", best_score)

# Create the Voting Regressor with the Best Weights
voting_model_optimized = VotingRegressor(estimators, weights=best_weights)
voting_model_optimized.fit(X_train, y_train)

# Evaluate the Optimized Model
y_pred_voting_optimized = voting_model_optimized.predict(X_test)
print("Optimized Voting Regressor RMSE:", mean_squared_error(y_test, y_pred_voting_optimized, squared=False))
print("Optimized Voting Regressor R² Score:", r2_score(y_test, y_pred_voting_optimized)) """
#%%[markdown]
# Thw above chunk was writen estimaate weights for voting regresion but cound not use as it took a very long time and did not finish.
#%%
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# List of all models
models = [linear_model, rf_model, xgb_model, lasso_model, cat_model, ridge_model, svr_model, voting_model, stacking_regressor]
model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'Lasso', 'CatBoost', 'Ridge', 'SVM', 'Voting Regressor', 'Stacking Regressor']

# Evaluating all models
results = []
for model, name in zip(models, model_names):
    rmse, r2 = evaluate_model(model, X_test, y_test)
    results.append({'Model': name, 'RMSE': rmse, 'R² Score': r2})

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)
# %%

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Set up the matplotlib figure
plt.figure(figsize=(14, 6))

# Plot RMSE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title('Comparison of Model RMSE')
plt.xticks(rotation=45)
plt.ylabel('RMSE')
plt.xlabel('Model')

# Plot R² Score
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² Score', data=results_df)
plt.title('Comparison of Model R² Score')
plt.xticks(rotation=45)
plt.ylabel('R² Score')
plt.xlabel('Model')

plt.tight_layout()
plt.show()

#%%[markdown]
# Based on RMSE and R² Score, the Stacking Regressor is the best performing model, followed by the Random Forest. Models like Linear Regression, Lasso, and Ridge perform poorly on this dataset, as indicated by the high RMSE and low R² scores. The SVM performed the worst according to these results.

# %%
