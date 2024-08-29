import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Załaduj dane
df = pd.read_csv(r'C:\Users\marczap\OneDrive - P4 Sp.z.o.o\Dokumenty\MachineLearning\F1\SkryptyUzywaneDoBadan\Main\data\F1_FULL_DATA.csv')

# Przygotowanie danych
df['In_Points'] = df['FinishPosition'] <= 10
df['On_Podium'] = df['FinishPosition'] <= 3

# Wybieramy cechy i target
features = ['Driver', 'StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']
X = df[features]
y_classification_points = df['In_Points']
y_classification_podium = df['On_Podium']

# Podział na zestawy treningowe i testowe (80% train, 20% test)
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

# Zapisz modele do plików
with open('models/random_forest_classifier_points.pkl', 'wb') as file:
    pickle.dump(best_classifier_points, file)

with open('models/random_forest_classifier_podium.pkl', 'wb') as file:
    pickle.dump(best_classifier_podium, file)
