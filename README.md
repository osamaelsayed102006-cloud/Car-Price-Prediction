# 🚗 Car Price Prediction

A Machine Learning project that predicts the price of used cars based on their specifications using several regression algorithms and a complete preprocessing pipeline.

## 📌 Project Overview

This project focuses on predicting car prices in Egyptian Pounds (EGP) using a dataset containing different car specifications.

The project includes:

* Data Cleaning
* Exploratory Data Analysis
* Outlier Handling
* Feature Preprocessing
* One-Hot Encoding
* Feature Scaling
* Machine Learning Models
* Cross Validation
* Hyperparameter Tuning using GridSearchCV
* Model Evaluation
* Streamlit Deployment

## 📊 Dataset

The dataset contains information about cars and their prices.

### Features

* Brand
* Model
* Kilometers
* Year
* Fuel Type
* Transmission Type
* Engine Capacity (CC)
* Body Type

### Target

* Price_EGP

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

## 🔄 Machine Learning Workflow

The project follows these steps:

1. Load the dataset
2. Clean missing and invalid values
3. Remove duplicate records
4. Handle numerical outliers
5. Separate features and target
6. Split the data into training and testing sets
7. Apply OneHotEncoder to categorical features
8. Apply StandardScaler to numerical features
9. Build preprocessing and model pipelines
10. Train multiple regression models
11. Evaluate the models
12. Apply Cross Validation
13. Perform Hyperparameter Tuning using GridSearchCV
14. Select the final model
15. Save the final model using Joblib
16. Deploy the application using Streamlit

## 🤖 Models Tested

The following regression models were evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* AdaBoost Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* Support Vector Regression (SVR)

## 📈 Final Model Performance

After hyperparameter tuning, the final models were evaluated on the test set.

| Model             |     R² |     MAE |    RMSE |
| ----------------- | -----: | ------: | ------: |
| Decision Tree     | 0.7901 | 329,561 | 790,623 |
| Random Forest     | 0.8336 | 299,286 | 703,845 |
| Gradient Boosting | 0.9140 | 224,854 | 506,122 |
| XGBoost           | 0.9081 | 249,541 | 523,102 |

The final deployed model is the tuned **Gradient Boosting Regressor**.

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

## 🌐 Live Demo

🚀 **Car Price Prediction — Live Demo**

https://car-price-prediction-uhhacdkwsgvhfupw2egazm.streamlit.app/

The application is deployed using Streamlit and can be accessed directly through the live demo.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/osamaelsayed102006-cloud/Car-Price-Prediction.git
```

### 2. Open the project folder

```bash
cd Car-Price-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install the requirements

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

## 🚀 Application Features

The Streamlit application allows the user to select:

* Car Brand
* Car Model
* Engine Capacity
* Body Type
* Fuel Type
* Transmission Type
* Kilometers
* Manufacturing Year

The application then predicts the estimated car price in Egyptian Pounds.

The dropdown menus are dynamically filtered so that the available options depend on the previously selected car specifications.

## 🧪 Example

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

The model uses these features to generate an estimated price.

## 📦 Requirements

The project requires:

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

## 🔮 Future Improvements

Possible future improvements include:

* More extensive hyperparameter tuning
* Additional car datasets
* More advanced feature engineering
* Improved UI/UX
* Model monitoring
* Cloud-based model serving
* Additional regression algorithms

## 👨‍💻 Author

**Osama El Sayed**

AI / Machine Learning Student

GitHub:

https://github.com/osamaelsayed102006-cloud

## ✅ Project Status

**Completed — Machine Learning Model + Streamlit Web Application Deployed**

## 📄 License

This project is created for educational and portfolio purposes.
