import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Załaduj dane
df = pd.read_csv('data/F1_FULL_DATA.csv')

# Przygotowanie danych
df['In_Points'] = df['FinishPosition'] <= 10
df['On_Podium'] = df['FinishPosition'] <= 3

# Wybór torów
tracks = ['Monaco Grand Prix', 'Italian Grand Prix', 'Abu Dhabi Grand Prix']

# Funkcja do analizy ważności cech dla danego toru
def analyze_track(track):
    print(f"Analiza dla toru: {track}")
    track_data = df[df['GrandPrix'] == track]
    
    # Wybieramy cechy i target
    features = ['Driver', 'StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']
    X = track_data[features]

    # Dane dla regresji
    y_regression = track_data['FinishPosition']

    # Dane dla klasyfikacji
    y_classification_points = track_data['In_Points']
    y_classification_podium = track_data['On_Podium']

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
    _, _, y_train_points, y_test_points = train_test_split(X, y_classification_points, test_size=0.2, random_state=42)
    _, _, y_train_podium, y_test_podium = train_test_split(X, y_classification_podium, test_size=0.2, random_state=42)

    # Model Random Forest dla regresji
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train_reg)
    y_pred_reg = regressor.predict(X_test)

    # Ocena modelu regresji
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    print(f"Mean Squared Error (Regresja): {mse}")

    # Model Random Forest dla klasyfikacji (czy w punktach)
    classifier_points = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier_points.fit(X_train, y_train_points)
    y_pred_points = classifier_points.predict(X_test)

    # Ocena modelu klasyfikacji (czy w punktach)
    accuracy_points = accuracy_score(y_test_points, y_pred_points)
    print(f"Accuracy (czy w punktach): {accuracy_points}")

    # Model Random Forest dla klasyfikacji (czy na podium)
    classifier_podium = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier_podium.fit(X_train, y_train_podium)
    y_pred_podium = classifier_podium.predict(X_test)

    # Ocena modelu klasyfikacji (czy na podium)
    accuracy_podium = accuracy_score(y_test_podium, y_pred_podium)
    print(f"Accuracy (czy na podium): {accuracy_podium}")

    # Ważność cech
    importances_reg = regressor.feature_importances_
    importances_points = classifier_points.feature_importances_
    importances_podium = classifier_podium.feature_importances_

    # Wykres ważności cech
    def plot_feature_importances(importances, features, title):
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Ważność cech')
        plt.show()

    plot_feature_importances(importances_reg, features, f'Ważność cech (Regresja) - {track}')
    plot_feature_importances(importances_points, features, f'Ważność cech (Klasyfikacja - czy w punktach) - {track}')
    plot_feature_importances(importances_podium, features, f'Ważność cech (Klasyfikacja - czy na podium) - {track}')

# Analiza dla wybranych torów
for track in tracks:
    analyze_track(track)
