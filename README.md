# 🚗 Car Price Prediction

A Machine Learning web application that predicts the estimated price of used cars in Egypt based on their specifications and usage information.

The project uses Machine Learning regression models, preprocessing pipelines, cross-validation, hyperparameter tuning, and Streamlit to provide an interactive car price prediction application.

---

## 📌 Project Overview

The goal of this project is to build a Machine Learning model capable of predicting a car's price in **EGP** based on:

* Brand
* Model
* Kilometers
* Year
* Fuel Type
* Transmission Type
* Engine Capacity
* Body Type

The trained model is integrated into a Streamlit web application where users can select their car specifications and receive an estimated price.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

---

## 🧠 Machine Learning Workflow

### 1. Data Cleaning

The dataset was cleaned by:

* Handling missing values
* Converting `Kilometers` to numeric values
* Cleaning `Price_EGP`
* Filling missing categorical values
* Removing duplicate rows

### 2. Exploratory Data Analysis

The dataset was explored using:

* Histograms
* Count plots
* Scatter plots
* Correlation heatmap

### 3. Outlier Handling

IQR-based outlier capping was applied to numerical input features.

The target variable `Price_EGP` was not capped.

### 4. Feature Preprocessing

Categorical features were encoded using:

```python
OneHotEncoder(handle_unknown='ignore')
```

Numerical features were scaled using:

```python
StandardScaler()
```

Both preprocessing steps were combined using `ColumnTransformer`.

### 5. Machine Learning Pipelines

Scikit-learn `Pipeline` was used to combine preprocessing and model training.

The following regression models were evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* AdaBoost Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* Support Vector Regression (SVR)

### 6. Cross Validation

Cross-validation was used to evaluate model performance more reliably.

### 7. Hyperparameter Tuning

`GridSearchCV` was used to tune:

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### 8. Final Model

The final application uses a tuned **Gradient Boosting Regressor** pipeline.

The complete pipeline, including preprocessing and the model, was saved using Joblib.

```python
joblib.dump(final_model, 'final_model.pkl')
```

---

## 📊 Model Performance

The tuned models were evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

| Model             |         R² |             MAE |            RMSE |
| ----------------- | ---------: | --------------: | --------------: |
| Decision Tree     |     0.7901 |     329,561 EGP |     790,623 EGP |
| Random Forest     |     0.8336 |     299,286 EGP |     703,845 EGP |
| Gradient Boosting | **0.9140** | **224,854 EGP** | **506,122 EGP** |
| XGBoost           |     0.9081 |     249,541 EGP |     523,102 EGP |

The Gradient Boosting pipeline was used as the final model for the Streamlit application based on its performance on the test set.

---

## 📁 Project Structure

```text
Car-Price-Prediction/
│
├── data/
│   └── car_ads_details_kaggle.csv
│
├── Main/
│   └── main.py
│
├── models/
│   └── final_model.pkl
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/osamaelsayed102006-cloud/Car-Price-Prediction.git
```

### 2. Navigate to the project

```bash
cd Car-Price-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

---

## 🌐 Live Demo

🚀 **[Car Price Prediction — Live Demo](YOUR_STREAMLIT_APP_URL)**

The application is deployed using Streamlit and can be accessed directly through the live demo.

---

## 🎯 Application Features

The Streamlit application provides:

* 🚗 Interactive car selection
* 🔗 Dependent dropdown menus
* ⚙️ Automatic feature selection
* 💰 Car price prediction
* 📋 Selected car details
* ⚡ Real-time predictions

The dropdowns follow a dependent selection flow:

```text
Brand
  ↓
Model
  ↓
Engine Capacity
  ↓
Body Type
  ↓
Fuel Type
  ↓
Transmission Type
```

This helps users select combinations based on the available dataset.

---

## 📈 Example

Example input:

```text
Brand: BMW
Model: 218 i
Engine Capacity: 1500 CC
Body Type: Sedan
Fuel Type: Benzine
Transmission: Automatic
Kilometers: 50,000 KM
Year: 2020
```

The application uses the trained Machine Learning pipeline to generate an estimated price in Egyptian Pounds.

---

## 📦 Requirements

The project dependencies are listed in `requirements.txt`.

Main libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
streamlit
joblib
```

---

## 🔮 Future Improvements

Possible future improvements include:

* Improving model performance
* Adding more vehicle features
* Expanding the dataset
* Improving the Streamlit interface
* Adding interactive visualizations
* Exploring additional Machine Learning techniques

---

## 👨‍💻 Author

**Osama Elsayed**

AI / Machine Learning Student

Interested in:

* Machine Learning
* Artificial Intelligence
* Python
* Backend Development

---

## 📌 Project Status

**Completed — Deployed Machine Learning Web Application**

The Machine Learning pipeline has been trained, evaluated, tuned, saved, and integrated into a Streamlit web application.

---

## 📄 License

This project is intended for educational and portfolio purposes.
