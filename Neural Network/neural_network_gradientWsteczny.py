import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Wczytanie danych z pliku CSV
data = pd.read_csv('data/F1_FULL_DATA.csv')

# Przygotowanie danych do modelowania
features = data[['StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']]
target = data['FinishPosition']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Budowa sieci neuronowej
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Trenowanie modelu
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)

# Dokonanie predykcji na zbiorze testowym
predictions = model.predict(X_test_scaled).flatten()

# Ocena modelu
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"R² score: {r2}")
print(f"Mean Absolute Error: {mae}")

# Funkcja do obliczania gradientów wstecznych
def get_feature_importances(model, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
    
    gradients = tape.gradient(predictions, inputs)
    
    # Oblicz średnią wartość gradientów dla każdej cechy
    mean_gradients = np.mean(np.abs(gradients), axis=0)
    
    return mean_gradients

# Obliczanie gradientów dla cech
importances = get_feature_importances(model, X_test_scaled)

# Wyświetlenie ważności cech
feature_names = features.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances based on Gradients:")
print(importance_df)

# Wykres ważności cech
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances for Predicting Finish Position based on Gradients')
plt.gca().invert_yaxis()
plt.show()
