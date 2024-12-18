# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.fft import fft
# import numpy as np

# # Charger les données
# # file_path = 'Instance_sample_dataset_v3/metadata/metadata_Instance_events_10k.csv'
# file_path = 'Instance_sample_dataset_v3/metadata/metadata_Instance_events_v3.csv'
# data = pd.read_csv(file_path)

# # 1. Répartition géographique des stations sismiques
# plt.figure(figsize=(10, 6))
# plt.scatter(data['station_longitude_deg'], data['station_latitude_deg'], alpha=0.5, c='blue', marker='o')
# plt.title('Répartition géographique des stations sismiques')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# Charger les données
file_path = '../Instance_sample_dataset_v3/metadata/metadata_Instance_events_10k.csv'
data = pd.read_csv(file_path)

# Normalisation des données
signal_data = data['trace_pgv_cmps'].dropna().values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# Préparation des séquences
sequence_length = 50
X, y = [], []
for i in range(len(signal_data) - sequence_length):
    X.append(signal_data[i:i+sequence_length])
    y.append(signal_data[i+sequence_length])
X, y = np.array(X), np.array(y)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Construction du modèle LSTM
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Fonction de perte personnalisée
def custom_loss(y_true, y_pred):
    weights = tf.where(tf.greater(y_true, 0), 2.0, 1.0)  # Double poids pour valeurs non nulles
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

# Compilation
model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Entraînement
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping]
)

# Prédictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Évaluation
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, color='blue', label='Valeurs réelles')
plt.plot(predictions, color='red', label='Prédictions LSTM')
plt.title("Prédiction des signaux sismiques avec LSTM")
plt.xlabel("Index")
plt.ylabel("Intensité (cm/s)")
plt.legend()
plt.show()