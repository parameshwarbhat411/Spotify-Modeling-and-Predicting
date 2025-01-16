# 🎵 Spotify Dataset Modeling and Prediction 🎶

## 📌 Project Overview

This project aims to **predict the popularity of songs** in the Spotify dataset based on their **musical attributes**. It explores **various machine learning models** to analyze and predict **popularity, danceability, and instrumentalness**, as well as classify **song genres**.

Through **exploratory data analysis (EDA)** and **predictive modeling**, we identify key patterns in the data and optimize models for **better forecasting accuracy**.

---

## 📊 Dataset Overview

- **Total Records:** 114,000  
- **Features:** 21  
- **Key Attributes:**
  - **Track Information:** Track ID, Artists, Album Name, Track Name
  - **Popularity Metrics:** Popularity Score (0-100)
  - **Musical Features:**
    - Danceability, Energy, Key, Loudness, Mode
    - Speechiness, Acousticness, Instrumentalness, Liveness, Valence
    - Tempo, Time Signature, Track Genre
  - **Content Features:** Explicit Content Flag, Duration

---

## 🔎 Exploratory Data Analysis (EDA)

### **1️⃣ Popularity Analysis**
- Examined **relationships** between **popularity** and **musical attributes**.
- Identified **features most correlated** with popularity.
- Explored **trends across different genres**.

### **2️⃣ Danceability Analysis**
- Investigated how **danceability** varies across **songs** and **genres**.
- Identified **factors impacting danceability**, such as:
  - **Mode**
  - **Valence**
  - **Energy**
  - **Speechiness**

### **3️⃣ Instrumentalness Analysis**
- Analyzed the **distribution** of **instrumental songs**.
- Explored **instrumentalness by genre** and its relationship with:
  - **Loudness**
  - **Energy**
  - **Acousticness**
- Found significant differences between **electronic vs. acoustic genres**.

---

## ⚙️ Machine Learning Models

We experimented with multiple **regression and classification models** to predict **popularity, danceability, and instrumentalness**, as well as classify **genres**.

### **📌 Models Used:**
- **Linear Regression**
- **Random Forest**
- **XGBoost**
- **Lasso & Ridge Regression**
- **Support Vector Machines (SVM)**
- **Voting & Stacking Regressors**

---

## 📈 Model Performance Results

### **1️⃣ Popularity Prediction**
| Model                | RMSE  | R² Score |
|----------------------|-------|---------|
| **Stacking Regressor** | **15.1677** | **0.5413** |
| Random Forest       | 16.2502 | 0.5121 |
| XGBoost            | 15.9003 | 0.5289 |

- **Best Model:** 📊 **Stacking Regressor**
- **Insights:** Feature importance analysis showed that **energy, valence, and danceability** had strong correlations with **popularity**.

---

### **2️⃣ Danceability Prediction**
| Model                          | RMSE  | R² Score |
|--------------------------------|-------|---------|
| **Tuned Random Forest Regressor** | **0.0087** | **0.7094** |
| XGBoost                       | 0.0102 | 0.6783 |
| Support Vector Regression      | 0.0121 | 0.6420 |

- **Best Model:** 🎶 **Tuned Random Forest Regressor**
- **Insights:** Higher **energy** and **valence** tend to result in **higher danceability**.

---

### **3️⃣ Instrumentalness Prediction**
| Model                | MSE   | R² Score |
|----------------------|-------|---------|
| **Random Forest**   | **0.037** | **0.615** |
| XGBoost            | 0.042  | 0.583  |
| Ridge Regression   | 0.048  | 0.545  |

- **Best Model:** 🎼 **Random Forest Regressor**
- **Insights:** Highly instrumental tracks often have **low speechiness** and **high acousticness**.

---

### **4️⃣ Genre Classification**
| Genre Type       | Overall Accuracy | Training Accuracy | Test Accuracy |
|-----------------|-----------------|-----------------|--------------|
| **Similar Genres** | **0.74** | 0.99 | 0.50 |
| **Distinct Genres** | **0.88** | 1.00 | 0.76 |

- **Best Model:** 🎯 **Random Forest Classifier**
- **Insights:** Classifying **similar genres** (e.g., pop vs. indie pop) is **challenging**, while **distinct genres** (e.g., classical vs. hip-hop) show **higher classification accuracy**.

---

## 🔮 Conclusion

- **Machine learning models effectively predict song popularity, danceability, and instrumentalness based on musical attributes.**  
- **Stacking Regressor** provided the **best performance for popularity prediction**.  
- **Tuned Random Forest** was **most effective for danceability and instrumentalness prediction**.  
- **Genre classification accuracy improves significantly when classifying distinct genres rather than similar ones.**  
- The results highlight that **certain musical attributes (e.g., energy, loudness, acousticness)** play a **crucial role in song characteristics**.

---

## 🚀 Next Steps

- **📌 Further Optimize Models:**  
  - Hyperparameter tuning for **XGBoost** and **Stacking models**.
  - Experiment with **Deep Learning models (LSTMs/Neural Networks)**.

- **🎵 Build a Song Recommendation System:**  
  - Leverage **collaborative filtering** and **content-based filtering**.
  - Utilize **user listening history** and **feature-based song similarity**.

- **📊 Deploy Interactive Dashboard:**  
  - Create a **Spotify Insights Dashboard** using **Streamlit or Flask**.
  - Allow users to **input song features** and get **predicted popularity & danceability**.

---

## 🏗️ Setup & Usage

### **1️⃣ Installation**
Clone the repository and install dependencies:

```sh
git clone https://github.com/your-username/spotify-analysis.git
cd spotify-analysis
pip install -r requirements.txt
