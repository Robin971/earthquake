import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = '../Instance_sample_dataset_v3/metadata/sorted_metadata_Instance_events_filtered.xlsx'
data = pd.read_excel(file_path)

data['source_origin_time'] = pd.to_datetime(data['source_origin_time'])
data = data.dropna()

features = [
    'source_latitude_deg',
    'source_longitude_deg',
    'source_depth_km',
    'source_magnitude',
    'path_travel_time_P_s',
    'path_travel_time_S_s',
    'path_ep_distance_km',
    'path_hyp_distance_km',
    'trace_pga_cmps2',
    'trace_pgv_cmps',
    'trace_sa10_cmps2',
    'trace_sa30_cmps2'
]

target = 'source_magnitude'

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])
scaled_target = scaler.fit_transform(data[[target]])

sequence_length = 10

def create_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, scaled_target, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(X_test)

y_test_rescaled = scaler.inverse_transform(y_test)
predictions_rescaled = scaler.inverse_transform(predictions)

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual Seismic Magnitude', marker='o')
plt.plot(predictions_rescaled, label='Predicted Seismic Magnitude', marker='x')
plt.title('Predicted vs Actual Seismic Magnitudes')
plt.xlabel('Time Step')
plt.ylabel('Magnitude')
plt.legend()
plt.show()

model.save('lstm_seismic_model.h5')
print("Model training complete. Model saved as 'lstm_seismic_model.h5'.")