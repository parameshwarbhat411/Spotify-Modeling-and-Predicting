
#%%
#Import 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# %%
df =pd.read_csv('dataset.csv')
df.head(5) #Read Data first 5 rows
#Check for null values
df.isnull().sum()
#Remove NA values
df = df.dropna()
df.isnull().sum()
df = df.drop("Unnamed: 0", axis=1) #remove the unnamed column
#checking for duplicates
df.duplicated().sum()
#removing duplicates
df = df.drop_duplicates() #remove all duplicates
df.duplicated().sum()
df_genres = df.copy()
#%%
df_genres.head()
#Checking variable types prior to encoding to know the names of genres
data_types = df_genres.dtypes
#track_genre is a string with several categories

#View all of the genres present in the dataset
unique_explicit_values = df_genres['track_genre'].unique()
print(unique_explicit_values)

num_unique_genres = df_genres['track_genre'].nunique()
print(f'Number of unique genres: {num_unique_genres}')

#There are 114 unique genres 

#Count the occurrences of each genre in the 'track_genre' column
genre_counts = df_genres['track_genre'].value_counts()
# Count the number of genres with at least 1000 rows
num_genres_with_1000_rows = (genre_counts >= 1000).sum()
print(f"Number of genres with at least 1000 rows: {num_genres_with_1000_rows}")

#33 genres have 1000+  records 
# Filter genres with at least 1000 rows
genres_1000 = genre_counts[genre_counts >= 1000].index

# Display the selected genres
print("Selected genres with at least 1000 rows:")
print(genres_1000)
#Subset the dataframe based on genres with 1000+ 
df_selected = df_genres[df_genres['track_genre'].isin(genres_1000)]

#This analysis will examine a subset of genres, those with at least 1000 rows so that 
#there is enough training data for predictions 

# Selecting relevant genres:
#GROUP 1
#We would like to explore how spotify can tell the difference between
#similar genres, lets select electornic music that we might not know
#the specific differences between ourselves, and see if a model can classify them

#Select the electronic genres
genres_list1 = ['disco', 'electronic', 'industrial', 'techno', 'synth-pop', 'funk']
selected_genres1 = df_selected[df_selected['track_genre'].isin(genres_list1)]
df_selected_genres_shape1 = selected_genres1.shape
print(f"Shape of the selected genres DataFrame: {df_selected_genres_shape1}")
selected_genres1.head()

# Selecting relevant genres:
#GROUP 2
# Now taking a look at very distinct genres that we don't
# Expect to sound similar

#Select the distinct genres
genres_list2 = ['acoustic', 'metalcore', 'rock', 'techno', 'sad', 'reggae']
selected_genres2 = df_selected[df_selected['track_genre'].isin(genres_list2)]
df_selected_genres_shape2 = selected_genres2.shape
print(f"Shape of the selected genres DataFrame: {df_selected_genres_shape2}")
selected_genres2.head()

#Plot group 1 and group 2 central tendency, outliers, and distributions 

# GROUP 1 Box Plots to check the outliers
features_continuous_numerical = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

sns.set(style="whitegrid")

for feature in features_continuous_numerical:
    # GROUP 1
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='track_genre', y=feature, data=selected_genres1)
    plt.title(f'{feature} by Genre (Group 1)')
    plt.xlabel('Genre')
    plt.ylabel(feature)
    plt.show()

    # Calculate mean of the variable for each genre (Group 1)
    mean_values1 = selected_genres1.groupby('track_genre')[feature].mean()

    # Print mean values (Group 1)
    print(f"Mean {feature} by Genre (Group 1):")
    print(mean_values1)
    print('\n' + '-'*30 + '\n')  # Separator for better readability

    # GROUP 2
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='track_genre', y=feature, data=selected_genres2)
    plt.title(f'{feature} by Genre (Group 2)')
    plt.xlabel('Genre')
    plt.ylabel(feature)
    plt.show()

    # Calculate mean of the variable for each genre (Group 2)
    mean_values2 = selected_genres2.groupby('track_genre')[feature].mean()

    # Print mean values (Group 2)
    print(f"Mean {feature} by Genre (Group 2):")
    print(mean_values2)
    print('\n' + '-'*30 + '\n')  # Separator for better readability

    # Compare means using a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mean_values1.index, y=mean_values1, color='blue', label='Group 1')
    sns.barplot(x=mean_values2.index, y=mean_values2, color='orange', label='Group 2')
    plt.title(f'Mean {feature} by Genre Comparison')
    plt.xlabel('Genre')
    plt.ylabel(f'Mean {feature}')
    plt.legend()
    plt.show()

features_continuous_numerical = ['popularity', 'duration_ms', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

sns.set(style="whitegrid")

for feature in features_continuous_numerical:
    # Calculate mean of the variable for each genre (Group 1)
    mean_values1 = selected_genres1.groupby('track_genre')[feature].mean()

    # Calculate mean of the variable for each genre (Group 2)
    mean_values2 = selected_genres2.groupby('track_genre')[feature].mean()

    # Plot GROUP 1 Pie Chart
    plt.figure(figsize=(8, 4))
    plt.pie(mean_values1, labels=mean_values1.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'), wedgeprops=dict(width=0.4))
    plt.title(f'Mean {feature} Distribution (Group 1)')
    plt.show()

    # Plot GROUP 2 Pie Chart
    plt.figure(figsize=(8, 4))
    plt.pie(mean_values2, labels=mean_values2.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'), wedgeprops=dict(width=0.4))
    plt.title(f'Mean {feature} Distribution (Group 2)')
    plt.show()


#%%
#######################################
#Initial Modeling - Decision Trees    #
#######################################

#%%

#PREDICT GENRES OVERALL - all genres 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Clean Data
music_data = df_genres
music_data.dropna(inplace=True)
music_data.drop_duplicates(inplace=True)

X = music_data.drop(columns = ["track_id", "artists", "album_name", "track_name", "track_genre"])
y = music_data["track_genre"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
overall_accuracy = accuracy_score(y, model.predict(X))  # Predict on the entire dataset

# Print the results
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Overall Accuracy: {overall_accuracy:.2f}")

#%%
#Genre group 1 
X = selected_genres1.drop(columns = ["track_id", "artists", "album_name", "track_name", "track_genre"])
y = selected_genres1["track_genre"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

# Make predictions
train_predictions = model2.predict(X_train)
test_predictions = model2.predict(X_test)

# Calculate accuracies
train_accuracy1 = accuracy_score(y_train, train_predictions)
test_accuracy1= accuracy_score(y_test, test_predictions)
overall_accuracy1 = accuracy_score(y, model2.predict(X))  # Predict on the entire dataset

# Print the results
print(f"Training Accuracy Genres 1: {train_accuracy1:.2f}")
print(f"Test Accuracy Genres 1: {test_accuracy1:.2f}")
print(f"Overall Accuraccy Genres 1: {overall_accuracy1:.2f}")

#%%
#genre group 2
X = selected_genres2.drop(columns = ["track_id", "artists", "album_name", "track_name", "track_genre"])
y = selected_genres2["track_genre"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)

# Make predictions
train_predictions = model3.predict(X_train)
test_predictions = model3.predict(X_test)

# Calculate accuracies
train_accuracy2 = accuracy_score(y_train, train_predictions)
test_accuracy2 = accuracy_score(y_test, test_predictions)
overall_accuracy2 = accuracy_score(y, model3.predict(X))  # Predict on the entire dataset

# Print the results
print(f"Training Accuracy Genres 2: {train_accuracy2:.2f}")
print(f"Test Accuracy Genres 2: {test_accuracy2:.2f}")
print(f"Overall Accuracy Genres 2: {overall_accuracy2:.2f}")
# %%
#Now plotting model performance
# Accuracies
accuracies = [overall_accuracy, test_accuracy, overall_accuracy1, test_accuracy1, overall_accuracy2, test_accuracy2]

# Labels
labels = ['Overall (All Genres)', 'Test (All Genres)', 'Genres 1', 'Genres 1 Test', 'Genres 2', 'Genres 2 Test']

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color=['blue', 'blue', 'orange', 'orange', 'green', 'green'])
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy percentages
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.show()

# %%
###########################
### FEATURE SELECTION #####
###########################

#%%
#GENRE SET 1 - Similar Genres: 

#Lets select the relevant columns. We won't be needing to encode aritst name etc
column_names = selected_genres1.columns
print(column_names)
columns_to_select = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 
                     'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                     'valence', 'tempo', 'time_signature', 'track_genre'] 

predict_genre1 = selected_genres1[columns_to_select].copy()


#Identify numeric and categorical columns
numeric_cols = []
categorical_cols = []
for col in predict_genre1.columns:
    if predict_genre1[col].dtype == np.float64 or predict_genre1[col].dtype == np.int64:
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print('numeric columns:', numeric_cols)
print('Categorical columns:', categorical_cols)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

for col in categorical_cols:
    predict_genre1[col] = label_encoder.fit_transform(predict_genre1[col])

# Display the updated DataFrame
print(predict_genre1.head())


#Correlation analysis 
correlation_matrix = predict_genre1.corr()
target_correlation = correlation_matrix['track_genre'].abs() 

# Select features with high correlation
selected_features = target_correlation[target_correlation > 0.2].index  # Adjust the correlation threshold as needed

print(selected_features)

#from this we get: 'duration_ms', 'instrumentalness', 'valence'

# Recursive Feature Elimination (RFE):
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = predict_genre1.drop('track_genre', axis=1) 
y = predict_genre1['track_genre']

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)  
fit = rfe.fit(X, y)

selected_features2 = X.columns[fit.support_]
print(selected_features2)

#from this we get: 'popularity', 'duration_ms', 'key', 'loudness', 'tempo'

# Feature Importance from Tree-based Models:
from sklearn.ensemble import RandomForestClassifier

X = predict_genre1.drop('track_genre', axis=1)  
y = predict_genre1['track_genre']

model = RandomForestClassifier()
model.fit(X, y)

feature_importance = model.feature_importances_
selected_features3 = X.columns[feature_importance > 0.10] 

print(selected_features3)
#from this we get: 'popularity', 'acousticness', 'instrumentalness'

# so our different kinds of feature selection method got us 3 different combos 
print("Correlation Matrix for Genres 1:", selected_features3)

print("RFE for Genres 1:", selected_features2)

print("Tree-based for Genres 1:", selected_features3)



# %%
#########
#GENRE SET 2 - Distinct Genres: 

#Lets select the relevant columns. We won't be needing to encode aritst name etc
column_names = selected_genres2.columns
print(column_names)
columns_to_select = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 
                     'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                     'valence', 'tempo', 'time_signature', 'track_genre'] 

predict_genre2 = selected_genres2[columns_to_select].copy()


#Identify numeric and categorical columns
numeric_cols = []
categorical_cols = []
for col in predict_genre2.columns:
    if predict_genre2[col].dtype == np.float64 or predict_genre2[col].dtype == np.int64:
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print('numeric columns:', numeric_cols)
print('Categorical columns:', categorical_cols)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

for col in categorical_cols:
    predict_genre2[col] = label_encoder.fit_transform(predict_genre2[col])

#Correlation analysis 
correlation_matrix = predict_genre2.corr()
target_correlation = correlation_matrix['track_genre'].abs()  

# Select features with high correlation
selected_features = target_correlation[target_correlation > 0.2].index  

print(selected_features)

#from this we get: 'duration_ms', 'instrumentalness', 'valence'

# Recursive Feature Elimination (RFE):
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = predict_genre2.drop('track_genre', axis=1) 
y = predict_genre2['track_genre']

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)  
fit = rfe.fit(X, y)

selected_features2 = X.columns[fit.support_]
print(selected_features2)

#from this we get: 'popularity', 'duration_ms', 'key', 'loudness', 'tempo'

# Feature Importance from Tree-based Models:
from sklearn.ensemble import RandomForestClassifier

X = predict_genre2.drop('track_genre', axis=1)  
y = predict_genre2['track_genre']

model = RandomForestClassifier()
model.fit(X, y)

feature_importance = model.feature_importances_
selected_features3 = X.columns[feature_importance > 0.10] 

print(selected_features3)
#from this we get: 'popularity', 'acousticness', 'instrumentalness'

# so our different kinds of feature selection method got us 3 different combos 
print("Correlation Matrix for Genres 2:", selected_features3)

print("RFE for Genres 2:", selected_features2)

print("Tree-based for Genres 2:", selected_features3)

# %%
#################################
#       Classifier Models       #
#################################
#%%
####GENRE SET 1########

music_data = selected_genres1
features_set1 = ['popularity', 'acousticness', 'instrumentalness']

X = music_data[features_set1]
y = music_data['track_genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f'Decision Tree Classifier Accuracy: {dt_accuracy:.4f}')

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Classifier Accuracy: {rf_accuracy:.4f}')

# Support Vector Classifier (SVC)
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
svc_predictions = svc_classifier.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f'Support Vector Classifier Accuracy: {svc_accuracy:.4f}')

# K-Nearest Neighbors Classifier (KNN)
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f'K-Nearest Neighbors Classifier Accuracy: {knn_accuracy:.4f}')
# %%
##### GENRE SET 2 #####
music_data = selected_genres2
features_set2 = ['popularity', 'danceability', 'energy', 'acousticness']

X = music_data[features_set2]
y = music_data['track_genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier2 = DecisionTreeClassifier()
dt_classifier2.fit(X_train, y_train)
dt_predictions2 = dt_classifier2.predict(X_test)
dt_accuracy2 = accuracy_score(y_test, dt_predictions2)
print(f'Decision Tree Classifier Accuracy: {dt_accuracy2:.4f}')

# Random Forest Classifier
rf_classifier2 = RandomForestClassifier()
rf_classifier2.fit(X_train, y_train)
rf_predictions2 = rf_classifier2.predict(X_test)
rf_accuracy2 = accuracy_score(y_test, rf_predictions2)
print(f'Random Forest Classifier Accuracy: {rf_accuracy2:.4f}')

# Support Vector Classifier (SVC)
svc_classifier2 = SVC()
svc_classifier2.fit(X_train, y_train)
svc_predictions2 = svc_classifier2.predict(X_test)
svc_accuracy2 = accuracy_score(y_test, svc_predictions2)
print(f'Support Vector Classifier Accuracy: {svc_accuracy2:.4f}')

# K-Nearest Neighbors Classifier (KNN)
knn_classifier2 = KNeighborsClassifier()
knn_classifier2.fit(X_train, y_train)
knn_predictions2 = knn_classifier2.predict(X_test)
knn_accuracy2 = accuracy_score(y_test, knn_predictions2)
print(f'K-Nearest Neighbors Classifier Accuracy: {knn_accuracy2:.4f}')

# %%
# Comparing Accurcaies
# Accuracies for Genre Set 1
accuracies_set1 = [dt_accuracy, rf_accuracy, svc_accuracy, knn_accuracy]

# Accuracies for Genre Set 2
accuracies_set2 = [dt_accuracy2, rf_accuracy2, svc_accuracy2, knn_accuracy2]

# Labels
labels = ['Decision Tree', 'Random Forest', 'SVC', 'KNN']

# Plotting
width = 0.35  
ind = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(ind - width/2, accuracies_set1, width, label='Similar Genres')
rects2 = ax.bar(ind + width/2, accuracies_set2, width, label='Distinct Genres')

ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracies by Genre Set')
ax.set_xticks(ind)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# %%
#######################################
#       Random Forests Deep Dive      #
#######################################
#%%
from sklearn.model_selection import GridSearchCV

# Features and labels for Genre Set 2
X = selected_genres2[features_set2]
y = selected_genres2['track_genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model
rf_classifier = RandomForestClassifier()

# Define the hyperparameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20],      # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

# Perform GridSearchCV
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Evaluate the model with the best hyperparameters
best_rf_classifier = grid_search.best_estimator_
best_rf_predictions = best_rf_classifier.predict(X_test)
best_rf_accuracy = accuracy_score(y_test, best_rf_predictions)
print(f'Random Forest Classifier Accuracy (Tuned): {best_rf_accuracy:.4f}')
train_predictions = best_rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f'Random Forest Classifier Train Accuracy (Tuned): {train_accuracy:.4f}')
best_rf_predictions = best_rf_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, best_rf_predictions)
print(f'Random Forest Classifier Test Accuracy (Tuned): {test_accuracy:.4f}')
# %%
#Compare to oringal test, train, and overall accuracy
music_data = selected_genres2
features_set2 = ['popularity', 'danceability', 'energy', 'acousticness']

X = music_data[features_set2]
y = music_data['track_genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier2 = RandomForestClassifier()
rf_classifier2.fit(X_train, y_train)
rf_predictions2 = rf_classifier2.predict(X_test)
rf_accuracy2 = accuracy_score(y_test, rf_predictions2)

# Predictions and Accuracies
train_predictions2 = rf_classifier2.predict(X_train)
test_predictions2 = rf_classifier2.predict(X_test)

train_accuracy2 = accuracy_score(y_train, train_predictions2)
test_accuracy2 = accuracy_score(y_test, test_predictions2)

# Print out accuracies
print(f'Random Forest Classifier Accuracy: {rf_accuracy2:.4f}')
print(f'Random Forest Classifier Train Accuracy: {train_accuracy2:.4f}')
print(f'Random Forest Classifier Test Accuracy: {test_accuracy2:.4f}')
