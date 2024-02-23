
# %%
####################################
#       Instrumentalness EDA            #
####################################


import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of 'instrumentalness'
plt.figure(figsize=(10, 6))
sns.histplot(df['instrumentalness'], bins=30, kde=True)
plt.title('Distribution of Instrumentalness')
plt.xlabel('Instrumentalness')
plt.ylabel('Frequency')
plt.show()

#The plot is a histogram of instrumentality, which is a measure of how much a song is dominated by instruments, as opposed to vocals. The x-axis of the plot is instrumentality, ranging from 0 to 1. The y-axis is the frequency of songs with that instrumentality, measured in counts.
#The plot shows that the distribution of instrumentality is bimodal, with two peaks at around 0.2 and 0.8. This means that there are two main groups of songs: those that are very vocal-driven, and those that are very instrumental-driven. There are also a significant number of songs in between these two peaks.

#%%
#Use box plots to visualize how instrumentalness varies across different genres or other categorical variables.
# Box plot for 'instrumentalness' by 'genre'
plt.figure(figsize=(12, 6))
sns.boxplot(x='track_genre', y='instrumentalness', data=df)
plt.title('Box Plot of Instrumentalness by Genre')
plt.xlabel('Genre')
plt.ylabel('Instrumentalness')
plt.xticks(rotation=45, ha='right')
plt.show()

#The box plot shows that there is a variation in instrumentality across genres. 
#Classical music songs are generally more instrumental-driven than pop music songs. The distribution of instrumentalities within each genre is also different. Classical music songs tend to have more similar instrumentalities than pop music songs.
# %%
# Distribution of 'instrumentalness' by 'genre'
plt.figure(figsize=(12, 6))
sns.histplot(x='instrumentalness', bins=30, kde=True, hue='track_genre', data=df)
plt.title('Distribution of Instrumentalness by Genre')
plt.xlabel('Instrumentalness')
plt.ylabel('Frequency')
plt.legend(title='Genre', loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()

#the graph shows that there is a wide range of instrumentality in songs across different genres. However, there are also some general trends, such as classical music being more instrumental-driven than pop music.
#Classical music songs are generally more instrumental-driven than pop music songs.
#Other genres, such as jazz and ambient music, also tend to have higher median instrumentalness than pop music.
#Hip hop and electronic dance music (EDM) tend to have lower median instrumentalness than pop music.

#%%
#the correlation between 'instrumentalness' and other numeric features.
# Correlation matrix focusing on 'instrumentalness'
correlation_instrumentalness = df.corr()['instrumentalness'].sort_values(ascending=False)
print("Correlation of 'instrumentalness' with other features:\n", correlation_instrumentalness)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


#the correlation matrix shows that there are some strong correlations between certain audio features, as well as some weaker correlations. 
#This information can be used to understand how different audio features relate to each other, and to create new features that are combinations of existing features.
#Danceability and energy have a strong positive correlation.
#Speechiness and acousticness have a strong negative correlation.
#Loudness and tempo have a weak positive correlation.
#Instrumentalness and valence have a weak negative correlation.

#%%
#scatter plots to visualize the relationship between 'instrumentalness' and other features.
# Scatter plot between 'instrumentalness' and 'energy'

plt.figure(figsize=(8, 6))
sns.scatterplot(x='instrumentalness', y='energy', data=df)
plt.title("Scatter Plot of 'instrumentalness' and 'energy'")
plt.show()
#There is a positive correlation between instrumentalness and energy, meaning that songs that are more instrumental tend to be more energetic.
#There is a wide range of instrumentalness and energy in music, with some songs being very instrumental and energetic, others being very instrumental but not energetic, and still others being vocal-driven and energetic.

#%%
#Explore how instrumentalness relates to popularity.

# Scatter plot between 'instrumentalness' and 'popularity'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='instrumentalness', y='popularity', data=df)
plt.title("Scatter Plot of 'instrumentalness' and 'popularity'")
plt.show()

#There is a negative correlation between instrumentalness and popularity, meaning that songs that are more instrumental are generally less popular.
#There is a wide range of instrumentality in popular songs, with some songs being very instrumental and others being very vocal-driven.
#the scatter plot shows that there is no one-size-fits-all answer to the question of whether instrumental songs can be popular. While instrumental songs are generally less popular than vocal-driven songs, there are still many popular instrumental songs out there.



#%%
#Linear Regressor

#import libraries needed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



# Features (X) and Target Variable (y)
X = df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1)
y = df['instrumentalness']

# Feature scaling (optional but recommended for Linear Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
linear_reg = LinearRegression()


# Train the model
linear_reg.fit(X_train, y_train)

# Make predictions
y_pred = linear_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Linear Regression - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

#*The Linear Regression model has an MSE of 0.0639, suggesting that, on average, the model's predictions deviate by this amount from the actual values. The R-squared value of 0.3163 indicates that the model explains about 31.63% of the variability in the target variable. 

# %%
#Decision Tree Regressor

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Features (X) and Target Variable (y)
X = df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1)
y = df['instrumentalness']

# Feature scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor model
decision_tree_reg = DecisionTreeRegressor(random_state=42)

# Train the model
decision_tree_reg.fit(X_train, y_train)



# Make predictions
y_pred = decision_tree_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Decision Tree Regressor - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")
# * The Decision Tree Regressor has a higher MSE compared to Linear Regression. In terms of MSE, lower values are preferred, so the Linear Regression model performed better in this regard.
# * The Decision Tree Regressor has a lower R² compared to Linear Regression. R² measures the proportion of variance explained, and a higher value is generally better. In this case, Linear Regression captured a larger proportion of the variance.
# * he Linear Regression model outperformed the Decision Tree Regressor in both MSE and R². It means that, based on the provided metrics, the Linear Regression model is preferable for this task.

#%%
#Gradient Boosting Modeling

from sklearn.ensemble import GradientBoostingRegressor


# Features (X) and Target Variable (y)
X = df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1)
y = df['instrumentalness']

# Feature scaling (optional but recommended for Gradient Boosting)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting Regressor model
gradient_boosting_reg = GradientBoostingRegressor(random_state=42)

# Train the model
gradient_boosting_reg.fit(X_train, y_train)


# Make predictions
y_pred = gradient_boosting_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Gradient Boosting Regressor - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

# * The Gradient Boosting Regressor has the lowest MSE among the three models. Lower MSE indicates better performance, so the Gradient Boosting Regressor outperforms both the Decision Tree Regressor and Linear Regression in terms of MSE.
# * The Gradient Boosting Regressor also has the highest R², indicating that it explains a larger proportion of the variance compared to the other models.
#The Gradient Boosting Regressor performed better than both the Decision Tree Regressor and Linear Regression in terms of both MSE and R².
#Gradient Boosting models are often powerful and can capture complex relationships in the data.
#%%

#Random forest Regressor

#Import library
from sklearn.ensemble import RandomForestRegressor


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1))
y = df['instrumentalness']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
random_forest_reg = RandomForestRegressor(random_state=42)

# Train the model
random_forest_reg.fit(X_train, y_train)

# Make predictions
y_pred = random_forest_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Random Forest Regressor - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

#*The Random Forest Regressor has the lowest MSE among all the models, indicating the best performance in terms of minimizing prediction errors.
#*The Random Forest Regressor also has the highest R², indicating that it explains a larger proportion of the variance compared to the other models.
#*The Random Forest Regressor outperforms the XGBoost Regressor, Gradient Boosting Regressor, and Decision Tree Regressor in terms of both MSE and R².

#%%
#XGBoost

from xgboost import XGBRegressor

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1))
y = df['instrumentalness']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgb_reg = XGBRegressor(random_state=42)

# Train the model
xgb_reg.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_reg.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Print evaluation metrics for XGBoost Regressor
print(f"XGBoost Regressor - Mean Squared Error: {mse_xgb:.4f}, R-squared: {r2_xgb:.4f}")
#*The XGBoost Regressor has a lower MSE than the Decision Tree Regressor and Gradient Boosting Regressor but a slightly higher MSE than the Random Forest Regressor.
#*The XGBoost Regressor has a higher R² than the Decision Tree Regressor and Gradient Boosting Regressor but a slightly lower R² than the Random Forest Regressor.
#*The XGBoost Regressor performs well, providing a good balance between MSE and R².

# %%
import pandas as pd

# Evaluate all models and store results
model_results = []

# Linear Regression
y_pred_linear = linear_reg.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
model_results.append({'Model': 'Linear Regression', 'Mean Squared Error': mse_linear, 'R-squared': r2_linear})

# Decision Tree Regressor
y_pred_decision_tree = decision_tree_reg.predict(X_test)
mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
r2_decision_tree = r2_score(y_test, y_pred_decision_tree)
model_results.append({'Model': 'Decision Tree Regressor', 'Mean Squared Error': mse_decision_tree, 'R-squared': r2_decision_tree})

# Gradient Boosting Regressor
y_pred_gradient_boosting = gradient_boosting_reg.predict(X_test)
mse_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting)
r2_gradient_boosting = r2_score(y_test, y_pred_gradient_boosting)
model_results.append({'Model': 'Gradient Boosting Regressor', 'Mean Squared Error': mse_gradient_boosting, 'R-squared': r2_gradient_boosting})

# Random Forest Regressor
y_pred_random_forest = random_forest_reg.predict(X_test)
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
r2_random_forest = r2_score(y_test, y_pred_random_forest)
model_results.append({'Model': 'Random Forest Regressor', 'Mean Squared Error': mse_random_forest, 'R-squared': r2_random_forest})

# XGBoost Regressor
y_pred_xgb = xgb_reg.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
model_results.append({'Model': 'XGBoost Regressor', 'Mean Squared Error': mse_xgb, 'R-squared': r2_xgb})

# Convert the results to a DataFrame
evaluation_df = pd.DataFrame(model_results)

# Display the DataFrame
print(evaluation_df)


# %%
# Calculate residuals
residuals = y_test - y_pred_random_forest

# Plotting the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Residuals for Random Forest Regressor')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#*The skewness and kurtosis values indicate that the distribution is not normally distributed. This is not uncommon for random forest regressors, as they can produce residuals with a variety of distributions.
#*Overall, the histogram suggests that the random forest regressor is performing well. The residuals are centered at zero and have a relatively small standard deviation. However, the skewed distribution and outliers suggest that there may be some outliers in the data that are affecting the model's predictions.

# %%
import matplotlib.pyplot as plt

# Plotting Mean Squared Error
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for Mean Squared Error
ax1.bar(evaluation_df['Model'], evaluation_df['Mean Squared Error'], color='blue', alpha=0.7)
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Model Mean Squared Error')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

# Plotting R-squared
fig, ax2 = plt.subplots(figsize=(10, 6))

# Line plot for R-squared
ax2.plot(evaluation_df['Model'], evaluation_df['R-squared'], marker='o', color='red')
ax2.set_ylabel('R-squared')
ax2.set_title('Model R-squared')


plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

#The bar graph shows the mean squared error (MSE) for four different machine learning models: linear regression, decision tree regressor, gradient boosting regressor, and random forest regressor. The MSE is a measure of how well a model's predictions match the actual values. A lower MSE indicates a better model fit.

#As you can see from the graph, the random forest regressor has the lowest MSE, followed by the gradient boosting regressor, the decision tree regressor, and finally, the linear regression model. This suggests that the random forest regressor is the best performing model out of the five.

#The Line graph shows that Random Forest Regressor is best fit compared to other model.A Higher R square indicates a better model fit.
#The random forest regressor is the best performing model out of the four, based on the MSE metric and R squared.
#%%
