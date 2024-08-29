import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Załaduj dane
df = pd.read_csv('data/F1_FULL_DATA.csv')

# Przygotowanie danych
df['In_Points'] = df['FinishPosition'] <= 10
df['On_Podium'] = df['FinishPosition'] <= 3

# Wybieramy cechy i target
features = ['Driver', 'StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']
X = df[features]
y_regression = df['FinishPosition']
y_classification_points = df['In_Points']
y_classification_podium = df['On_Podium']

# Podział na zestawy treningowe i testowe (80% train, 20% test)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_points, X_test_points, y_train_points, y_test_points = train_test_split(X, y_classification_points, test_size=0.2, random_state=42)
X_train_podium, X_test_podium, y_train_podium, y_test_podium = train_test_split(X, y_classification_podium, test_size=0.2, random_state=42)

# Definicja siatki hiperparametrów
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Ustawienie k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Optymalizacja modelu Random Forest dla regresji z k-fold cross-validation
regressor = RandomForestRegressor(random_state=42)
grid_search_reg = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_reg.fit(X_train_reg, y_train_reg)

# Najlepsze parametry dla regresji
best_params_reg = grid_search_reg.best_params_
print(f"Najlepsze parametry dla regresji: {best_params_reg}")

# Ocena najlepszego modelu regresji
best_regressor = grid_search_reg.best_estimator_
y_pred_reg = best_regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Mean Squared Error (Regresja): {mse}")

# Optymalizacja modelu Random Forest dla klasyfikacji (czy w punktach) z k-fold cross-validation
classifier_points = RandomForestClassifier(random_state=42)
grid_search_points = GridSearchCV(estimator=classifier_points, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_points.fit(X_train_points, y_train_points)

# Najlepsze parametry dla klasyfikacji (czy w punktach)
best_params_points = grid_search_points.best_params_
print(f"Najlepsze parametry dla klasyfikacji (czy w punktach): {best_params_points}")

# Ocena najlepszego modelu klasyfikacji (czy w punktach)
best_classifier_points = grid_search_points.best_estimator_
y_pred_points = best_classifier_points.predict(X_test_points)
accuracy_points = accuracy_score(y_test_points, y_pred_points)
print(f"Accuracy (czy w punktach): {accuracy_points}")

# Optymalizacja modelu Random Forest dla klasyfikacji (czy na podium) z k-fold cross-validation
classifier_podium = RandomForestClassifier(random_state=42)
grid_search_podium = GridSearchCV(estimator=classifier_podium, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_podium.fit(X_train_podium, y_train_podium)

# Najlepsze parametry dla klasyfikacji (czy na podium)
best_params_podium = grid_search_podium.best_params_
print(f"Najlepsze parametry dla klasyfikacji (czy na podium): {best_params_podium}")

# Ocena najlepszego modelu klasyfikacji (czy na podium)
best_classifier_podium = grid_search_podium.best_estimator_
y_pred_podium = best_classifier_podium.predict(X_test_podium)
accuracy_podium = accuracy_score(y_test_podium, y_pred_podium)
print(f"Accuracy (czy na podium): {accuracy_podium}")

# Ważność cech
importances_reg = best_regressor.feature_importances_
importances_points = best_classifier_points.feature_importances_
importances_podium = best_classifier_podium.feature_importances_

# Wykres ważności cech
def plot_feature_importances(importances, features, title):
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Ważność cech')
    plt.show()

plot_feature_importances(importances_reg, features, 'Ważność cech (Regresja)')
plot_feature_importances(importances_points, features, 'Ważność cech (Klasyfikacja - czy w punktach)')
plot_feature_importances(importances_podium, features, 'Ważność cech (Klasyfikacja - czy na podium)')
