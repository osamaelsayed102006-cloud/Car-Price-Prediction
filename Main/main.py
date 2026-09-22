import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)


# =========================================================
# Load Data
# =========================================================

df = pd.read_csv('data/car_ads_details_kaggle.csv')

print('========================================')
print('===== DATASET HEAD =====')
print('========================================')
print(df.head())

print('\n===== DATASET SHAPE =====')
print(df.shape)

print('\n===== DATASET INFO =====')
df.info()


# =========================================================
# Data Cleaning
# =========================================================

df['Kilometers'] = df['Kilometers'].replace('-', np.nan)

df['Kilometers'] = df['Kilometers'].str.replace(
    ',',
    ''
).astype(float)

df['Kilometers'] = df['Kilometers'].fillna(
    df['Kilometers'].median()
)


df['Price_EGP'] = df['Price_EGP'].str.replace(
    'EGP',
    ''
)

df['Price_EGP'] = df['Price_EGP'].str.replace(
    ',',
    ''
).astype(float)

df['Price_EGP'] = df['Price_EGP'].fillna(
    df['Price_EGP'].median()
)


df['Brand'] = df['Brand'].fillna(
    df['Brand'].mode()[0]
)

df['Model'] = df['Model'].fillna(
    df['Model'].mode()[0]
)

df['Engine Capacity (CC)'] = df['Engine Capacity (CC)'].fillna(
    df['Engine Capacity (CC)'].mode()[0]
)

df['Body Type'] = df['Body Type'].fillna(
    df['Body Type'].mode()[0]
)


print('\n========================================')
print('===== MISSING VALUES =====')
print('========================================')

print(df.isnull().sum())


print('\n===== DUPLICATES =====')

print(df.duplicated().sum())


df.drop_duplicates(
    inplace=True
)


print('\n===== DATASET INFO AFTER CLEANING =====')

df.info()


# =========================================================
# Outlier Capping
# =========================================================

Num_cols = df.drop(
    'Price_EGP',
    axis=1
).select_dtypes(
    include=np.number
).columns


for col in Num_cols:

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    df[col] = np.where(
        df[col] < lower_bound,
        lower_bound,
        np.where(
            df[col] > upper_bound,
            upper_bound,
            df[col]
        )
    )


print('\n========================================')
print('===== OUTLIER CAPPING DONE =====')
print('========================================')


# =========================================================
# EDA
# =========================================================

print('\n========================================')
print('===== EDA =====')
print('========================================')


for col in Num_cols:

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title(
        f'Distribution of {col}'
    )

    plt.show()


plt.figure(
    figsize=(12, 6)
)

df['Brand'].value_counts().head(15).plot(
    kind='bar'
)

plt.title(
    'Top 15 Car Brands'
)

plt.xlabel(
    'Brand'
)

plt.ylabel(
    'Count'
)

plt.xticks(
    rotation=45
)

plt.show()


plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x='Fuel Type'
)

plt.title(
    'Fuel Type Distribution'
)

plt.xticks(
    rotation=45
)

plt.show()


plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x='Transmission Type'
)

plt.title(
    'Transmission Type Distribution'
)

plt.xticks(
    rotation=45
)

plt.show()


plt.figure(
    figsize=(10, 5)
)

sns.countplot(
    data=df,
    x='Body Type'
)

plt.title(
    'Body Type Distribution'
)

plt.xticks(
    rotation=45
)

plt.show()


plt.figure(
    figsize=(8, 5)
)

sns.scatterplot(
    data=df,
    x='Kilometers',
    y='Price_EGP'
)

plt.title(
    'Kilometers vs Price'
)

plt.show()


plt.figure(
    figsize=(8, 5)
)

sns.scatterplot(
    data=df,
    x='Year',
    y='Price_EGP'
)

plt.title(
    'Year vs Price'
)

plt.show()


plt.figure(
    figsize=(8, 5)
)

sns.scatterplot(
    data=df,
    x='Engine Capacity (CC)',
    y='Price_EGP'
)

plt.title(
    'Engine Capacity vs Price'
)

plt.show()


corr = df.select_dtypes(
    include=np.number
).corr()


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title(
    'Correlation Matrix'
)

plt.show()


price_corr = corr[
    'Price_EGP'
].sort_values(
    ascending=False
)


print('\n===== PRICE CORRELATION =====')

print(price_corr)


# =========================================================
# Features / Target
# =========================================================

X = df.drop(
    'Price_EGP',
    axis=1
)

y = df['Price_EGP']


# =========================================================
# Columns
# =========================================================

categorical_cols = X.select_dtypes(include='str').columns
numerical_cols = X.select_dtypes(exclude='str').columns


print('\n========================================')
print('===== FEATURES =====')
print('========================================')

print('Categorical Columns:')
print(categorical_cols)

print('\nNumerical Columns:')
print(numerical_cols)


# =========================================================
# Preprocessing
# =========================================================

preprocessor = ColumnTransformer([

    (
        'cat',
        OneHotEncoder(
            handle_unknown='ignore'
        ),
        categorical_cols
    ),

    (
        'num',
        StandardScaler(),
        numerical_cols
    )
])


print('\n========================================')
print('===== PREPROCESSING READY =====')
print('========================================')


# =========================================================
# Train Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print('\n========================================')
print('===== TRAIN TEST SPLIT =====')
print('========================================')

print('X_train:', X_train.shape)

print('X_test:', X_test.shape)

print('y_train:', y_train.shape)

print('y_test:', y_test.shape)


# =========================================================
# Linear Regression
# =========================================================

lr_pipeline = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),

    (
        'model',
        LinearRegression()
    )
])


lr_pipeline.fit(
    X_train,
    y_train
)


lr_y_pred = lr_pipeline.predict(
    X_test
)


print('\n========================================')
print('===== LINEAR REGRESSION =====')
print('========================================')

print(
    'MSE:',
    mean_squared_error(
        y_test,
        lr_y_pred
    )
)

print(
    'R2:',
    r2_score(
        y_test,
        lr_y_pred
    )
)

print(
    'MAE:',
    mean_absolute_error(
        y_test,
        lr_y_pred
    )
)

print(
    'RMSE:',
    root_mean_squared_error(
        y_test,
        lr_y_pred
    )
)


# =========================================================
# Decision Tree
# =========================================================

dt_pipeline = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),

    (
        'model',
        DecisionTreeRegressor(
            random_state=42
        )
    )
])


dt_pipeline.fit(
    X_train,
    y_train
)


dt_y_pred = dt_pipeline.predict(
    X_test
)


print('\n========================================')
print('===== DECISION TREE =====')
print('========================================')

print(
    'MSE:',
    mean_squared_error(
        y_test,
        dt_y_pred
    )
)

print(
    'R2:',
    r2_score(
        y_test,
        dt_y_pred
    )
)

print(
    'MAE:',
    mean_absolute_error(
        y_test,
        dt_y_pred
    )
)

print(
    'RMSE:',
    root_mean_squared_error(
        y_test,
        dt_y_pred
    )
)


# =========================================================
# Random Forest
# =========================================================

rf_pipeline = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),

    (
        'model',
        RandomForestRegressor(
            random_state=42
        )
    )
])


rf_pipeline.fit(
    X_train,
    y_train
)


rf_y_pred = rf_pipeline.predict(
    X_test
)


print('\n========================================')
print('===== RANDOM FOREST =====')
print('========================================')

print(
    'MSE:',
    mean_squared_error(
        y_test,
        rf_y_pred
    )
)

print(
    'R2:',
    r2_score(
        y_test,
        rf_y_pred
    )
)

print(
    'MAE:',
    mean_absolute_error(
        y_test,
        rf_y_pred
    )
)

print(
    'RMSE:',
    root_mean_squared_error(
        y_test,
        rf_y_pred
    )
)


# =========================================================
# AdaBoost
# =========================================================

ada_pipeline = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),

    (
        'model',
        AdaBoostRegressor(
            random_state=42
        )
    )
])


ada_pipeline.fit(
    X_train,
    y_train
)


ada_y_pred = ada_pipeline.predict(
    X_test
)


print('\n========================================')
print('===== ADABOOST =====')
print('========================================')

print(
    'MSE:',
    mean_squared_error(
        y_test,
        ada_y_pred
    )
)

print(
    'R2:',
    r2_score(
        y_test,
        ada_y_pred
    )
)

print(
    'MAE:',
    mean_absolute_error(
        y_test,
        ada_y_pred
    )
)

print(
    'RMSE:',
    root_mean_squared_error(
        y_test,
        ada_y_pred
    )
)


# =========================================================
# Gradient Boosting
# =========================================================

gb_pipeline = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),

    (
        'model',
        GradientBoostingRegressor(
            random_state=42
        )
    )
])


gb_pipeline.fit(
    X_train,
    y_train
)


gb_y_pred = gb_pipeline.predict(
    X_test
)


print('\n========================================')
print('===== GRADIENT BOOSTING =====')
print('========================================')

print(
    'MSE:',
    mean_squared_error(
        y_test,
        gb_y_pred
    )
)

print(
    'R2:',
    r2_score(
        y_test,
        gb_y_pred
    )
)

print(
    'MAE:',
    mean_absolute_error(
        y_test,
        gb_y_pred
    )
)

print(
    'RMSE:',
    root_mean_squared_error(
        y_test,
        gb_y_pred
    )
)


# =========================================================
# XGBoost
# =========================================================

xgb_pipeline = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),

    (
        'model',
        XGBRegressor(
            random_state=42
        )
    )
])


xgb_pipeline.fit(
    X_train,
    y_train
)


xgb_y_pred = xgb_pipeline.predict(
    X_test
)


print('\n========================================')
print('===== XGBOOST =====')
print('========================================')

print(
    'MSE:',
    mean_squared_error(
        y_test,
        xgb_y_pred
    )
)

print(
    'R2:',
    r2_score(
        y_test,
        xgb_y_pred
    )
)

print(
    'MAE:',
    mean_absolute_error(
        y_test,
        xgb_y_pred
    )
)

print(
    'RMSE:',
    root_mean_squared_error(
        y_test,
        xgb_y_pred
    )
)


# =========================================================
# SVR
# =========================================================

svr_pipeline = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),

    (
        'model',
        SVR()
    )
])


svr_pipeline.fit(
    X_train,
    y_train
)


svr_y_pred = svr_pipeline.predict(
    X_test
)


print('\n========================================')
print('===== SVR =====')
print('========================================')

print(
    'MSE:',
    mean_squared_error(
        y_test,
        svr_y_pred
    )
)

print(
    'R2:',
    r2_score(
        y_test,
        svr_y_pred
    )
)

print(
    'MAE:',
    mean_absolute_error(
        y_test,
        svr_y_pred
    )
)

print(
    'RMSE:',
    root_mean_squared_error(
        y_test,
        svr_y_pred
    )
)


# =========================================================
# Cross Validation
# =========================================================

print('\n========================================')
print('===== CROSS VALIDATION =====')
print('========================================')


lr_scores = cross_val_score(
    lr_pipeline,
    X,
    y,
    cv=5,
    scoring='r2'
)

print('\n--- Linear Regression CV ---')

print('Scores:', lr_scores)

print('Mean Score:', lr_scores.mean())


dt_scores = cross_val_score(
    dt_pipeline,
    X,
    y,
    cv=5,
    scoring='r2'
)

print('\n--- Decision Tree CV ---')

print('Scores:', dt_scores)

print('Mean Score:', dt_scores.mean())


rf_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring='r2'
)

print('\n--- Random Forest CV ---')

print('Scores:', rf_scores)

print('Mean Score:', rf_scores.mean())


xgb_scores = cross_val_score(
    xgb_pipeline,
    X,
    y,
    cv=5,
    scoring='r2'
)

print('\n--- XGBoost CV ---')

print('Scores:', xgb_scores)

print('Mean Score:', xgb_scores.mean())


# =========================================================
# GridSearchCV
# =========================================================

print('\n========================================')
print('===== GRID SEARCH =====')
print('========================================')


# =========================================================
# XGBoost GridSearch
# =========================================================

xgb_param_grid = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__max_depth': [3, 4, 5, 6],
    'model__learning_rate': [0.05, 0.1]
}


xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)


xgb_grid.fit(
    X_train,
    y_train
)


print('\n===== XGBOOST GRID SEARCH =====')

print(
    'Best Parameters:',
    xgb_grid.best_params_
)

print(
    'Best CV Score:',
    xgb_grid.best_score_
)


# =========================================================
# Decision Tree GridSearch
# =========================================================

dt_param_grid = {
    'model__max_depth': range(3, 11),
    'model__min_samples_split': range(2, 11),
    'model__min_samples_leaf': range(1, 6)
}


dt_grid = GridSearchCV(
    dt_pipeline,
    dt_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)


dt_grid.fit(
    X_train,
    y_train
)


print('\n===== DECISION TREE GRID SEARCH =====')

print(
    'Best Parameters:',
    dt_grid.best_params_
)

print(
    'Best CV Score:',
    dt_grid.best_score_
)


# =========================================================
# Random Forest GridSearch
# =========================================================

rf_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}


rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)


rf_grid.fit(
    X_train,
    y_train
)


print('\n===== RANDOM FOREST GRID SEARCH =====')

print(
    'Best Parameters:',
    rf_grid.best_params_
)

print(
    'Best CV Score:',
    rf_grid.best_score_
)


# =========================================================
# Gradient Boosting GridSearch
# =========================================================

gb_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1]
}


gb_grid = GridSearchCV(
    gb_pipeline,
    gb_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)


gb_grid.fit(
    X_train,
    y_train
)


print('\n===== GRADIENT BOOSTING GRID SEARCH =====')

print(
    'Best Parameters:',
    gb_grid.best_params_
)

print(
    'Best CV Score:',
    gb_grid.best_score_
)


# =========================================================
# Final Evaluation
# =========================================================

print('\n========================================')
print('===== FINAL EVALUATION =====')
print('========================================')


# =========================================================
# Decision Tree Final
# =========================================================

dt_best_pred = dt_grid.predict(
    X_test
)


dt_final_mse = mean_squared_error(
    y_test,
    dt_best_pred
)

dt_final_r2 = r2_score(
    y_test,
    dt_best_pred
)

dt_final_mae = mean_absolute_error(
    y_test,
    dt_best_pred
)

dt_final_rmse = root_mean_squared_error(
    y_test,
    dt_best_pred
)


print('\n===== DECISION TREE FINAL =====')

print('MSE:', dt_final_mse)

print('R2:', dt_final_r2)

print('MAE:', dt_final_mae)

print('RMSE:', dt_final_rmse)


# =========================================================
# Random Forest Final
# =========================================================

rf_best_pred = rf_grid.predict(
    X_test
)


rf_final_mse = mean_squared_error(
    y_test,
    rf_best_pred
)

rf_final_r2 = r2_score(
    y_test,
    rf_best_pred
)

rf_final_mae = mean_absolute_error(
    y_test,
    rf_best_pred
)

rf_final_rmse = root_mean_squared_error(
    y_test,
    rf_best_pred
)


print('\n===== RANDOM FOREST FINAL =====')

print('MSE:', rf_final_mse)

print('R2:', rf_final_r2)

print('MAE:', rf_final_mae)

print('RMSE:', rf_final_rmse)


# =========================================================
# Gradient Boosting Final
# =========================================================

gb_best_pred = gb_grid.predict(
    X_test
)


gb_final_mse = mean_squared_error(
    y_test,
    gb_best_pred
)

gb_final_r2 = r2_score(
    y_test,
    gb_best_pred
)

gb_final_mae = mean_absolute_error(
    y_test,
    gb_best_pred
)

gb_final_rmse = root_mean_squared_error(
    y_test,
    gb_best_pred
)


print('\n===== GRADIENT BOOSTING FINAL =====')

print('MSE:', gb_final_mse)

print('R2:', gb_final_r2)

print('MAE:', gb_final_mae)

print('RMSE:', gb_final_rmse)


# =========================================================
# XGBoost Final
# =========================================================

xgb_best_pred = xgb_grid.predict(
    X_test
)


xgb_final_mse = mean_squared_error(
    y_test,
    xgb_best_pred
)

xgb_final_r2 = r2_score(
    y_test,
    xgb_best_pred
)

xgb_final_mae = mean_absolute_error(
    y_test,
    xgb_best_pred
)

xgb_final_rmse = root_mean_squared_error(
    y_test,
    xgb_best_pred
)


print('\n===== XGBOOST FINAL =====')

print('MSE:', xgb_final_mse)

print('R2:', xgb_final_r2)

print('MAE:', xgb_final_mae)

print('RMSE:', xgb_final_rmse)


# =========================================================
# Final Comparison
# =========================================================

results = pd.DataFrame({

    'Model': [
        'Decision Tree',
        'Random Forest',
        'Gradient Boosting',
        'XGBoost'
    ],

    'R2': [
        dt_final_r2,
        rf_final_r2,
        gb_final_r2,
        xgb_final_r2
    ],

    'MAE': [
        dt_final_mae,
        rf_final_mae,
        gb_final_mae,
        xgb_final_mae
    ],

    'RMSE': [
        dt_final_rmse,
        rf_final_rmse,
        gb_final_rmse,
        xgb_final_rmse
    ],

    'MSE': [
        dt_final_mse,
        rf_final_mse,
        gb_final_mse,
        xgb_final_mse
    ]
})


print('\n========================================')
print('===== FINAL MODEL COMPARISON =====')
print('========================================\n')

print(
    results.sort_values(
        by='R2',
        ascending=False
    ).to_string(index=False)
)

# =========================================================
# Save Final Model
# =========================================================


final_model = gb_grid.best_estimator_

joblib.dump(
    final_model,
    'models/final_model.pkl'
)

print('\n========================================')
print('===== FINAL MODEL SAVED =====')
print('========================================')

print('Model saved as: final_model.pkl')