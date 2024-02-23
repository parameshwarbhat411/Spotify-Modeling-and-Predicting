# Spotify Dataset Modeling and Predicting

**Project Overview**

This project aims to predict the popularity of songs in the Spotify dataset based on their musical attributes. It explores various machine learning algorithms to model and predict the popularity, danceability, and instrumentalness of songs.

**Dataset**

The dataset contains 114,000 records with 21 features, including track ID, artists, album name, track name, popularity, duration, explicit content, and various musical attributes like danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time signature, and track genre.

**Exploratory Data Analysis (EDA)**

The EDA focuses on analyzing the distribution and correlation of numeric variables such as popularity, danceability, and instrumentalness. It also includes categorical analysis, particularly the exploration of different genres.

**Popularity Analysis**

Examined the relationship between popularity and other musical attributes.
Investigated which features are most correlated with popularity.

**Danceability Analysis**

Explored the distribution of danceability and its correlation with other features.
Evaluated the impact of mode, valence, energy, and speechiness on danceability.

**Instrumentalness Analysis**

Analyzed the distribution of instrumentalness and its relationship with other attributes.
Investigated how instrumentalness varies across different genres.

**Modeling**

Various regression and classification models were used to predict popularity, danceability, and instrumentalness, as well as to classify genres. The models include:

- Linear Regression
- Random Forest
- XGBoost
- Lasso
- Ridge
- SVM
- Voting Regressor
- Stacking Regressor
- Popularity Modeling Results

The Stacking Regressor showed the best performance with an RMSE of 15.1677 and an R² score of 0.5413.

**Danceability Modeling Results**

The Tuned Random Forest Regressor was the most effective model for predicting danceability, with the lowest RMSE of 0.0087 and the highest R² of 0.7094.

**Instrumentalness Modeling Results**

The Random Forest Regressor performed best in predicting instrumentalness, with an MSE of 0.037 and an R² of 0.615.

**Genre Classification Results**

For classifying similar genres, the overall accuracy was 0.74, with a training accuracy of 0.99 and a test accuracy of 0.50. For distinct genres, the overall accuracy was 0.88, with a training accuracy of 1.00 and a test accuracy of 0.76.

**Conclusion**

The project demonstrated that machine learning models could effectively predict song popularity, danceability, and instrumentalness based on musical attributes. The choice of model depends on specific requirements, such as the need for interpretability or predictive power.

**Next Steps**

Further tune the models to improve accuracy.
Develop a song recommendation algorithm based on the findings.
