import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load the data
file_path = 'data/F1_FULL_DATA.csv'
df = pd.read_csv(file_path)

# Przygotowanie cech (X) i etykiet (y)
X = df[['StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']].values
y = df['FinishPosition'].values

# Normalizacja danych
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Konwersja etykiet do formatu one-hot encoding
y = to_categorical(y - 1, num_classes=20)  # Zakładamy, że pozycje są od 1 do 20

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    model = Sequential([
        InputLayer(input_shape=(X_train.shape[1],)),  # Definicja kształtu danych wejściowych
        Dense(256, activation='relu'),               # Pierwsza warstwa ukryta z 256 neuronami i aktywacją ReLU
        Dropout(0.4),                                # Dropout dla regularizacji
        Dense(128, activation='relu'),               # Druga warstwa ukryta z 128 neuronami i aktywacją ReLU
        Dropout(0.4),                                # Dropout dla regularizacji
        Dense(20, activation='softmax')              # Warstwa wyjściowa z 20 neuronami (20 klas) z aktywacją softmax
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Tworzenie modelu
model = create_model()

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Trenowanie modelu
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Ocena modelu na zbiorze testowym
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Przewidywanie na zbiorze testowym
predictions = model.predict(X_test)

# Konwersja przewidywań i rzeczywistych wartości z one-hot encoding do klas
y_test_class = np.argmax(y_test, axis=1) + 1
predictions_class = np.argmax(predictions, axis=1) + 1

# Generowanie raportu klasyfikacji
report = classification_report(y_test_class, predictions_class, digits=4)
print(report)

# Przykładowe przewidywanie
example_input = np.array([[2, 1.0088770365603859, 1.005988580613432, 0.9927741124725101]])
example_input = scaler.transform(example_input)
example_output = model.predict(example_input)
predicted_position = np.argmax(example_output[0]) + 1  # Dodajemy 1, ponieważ klasy są od 0 do 19
print(f'Predicted Finish Position: {predicted_position}')