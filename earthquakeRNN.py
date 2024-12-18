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

location_targets = ['source_latitude_deg', 'source_longitude_deg']
magnitude_target = 'source_magnitude'

scaler_features = MinMaxScaler()
scaler_location = MinMaxScaler()
scaler_magnitude = MinMaxScaler()

scaled_data = scaler_features.fit_transform(data[features])
scaled_locations = scaler_location.fit_transform(data[location_targets])
scaled_magnitudes = scaler_magnitude.fit_transform(data[[magnitude_target]])

sequence_length = 10

def create_sequences(features, locations, magnitudes, sequence_length):
    X, loc, mag = [], [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        loc.append(locations[i + sequence_length])
        mag.append(magnitudes[i + sequence_length])
    return np.array(X), np.array(loc), np.array(mag)

X, y_locations, y_magnitudes = create_sequences(scaled_data, scaled_locations, scaled_magnitudes, sequence_length)

X_train, X_test, y_loc_train, y_loc_test, y_mag_train, y_mag_test = train_test_split(
    X, y_locations, y_magnitudes, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3)  
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train, np.hstack([y_loc_train, y_mag_train]), epochs=20, batch_size=32,
                    validation_data=(X_test, np.hstack([y_loc_test, y_mag_test])))

test_loss, test_mae = model.evaluate(X_test, np.hstack([y_loc_test, y_mag_test]))
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs (RNN)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

training_accuracy = 100 - (np.array(history.history['mae']) * 100)
validation_accuracy = 100 - (np.array(history.history['val_mae']) * 100)

plt.figure(figsize=(12, 6))
plt.plot(training_accuracy, label='Training Accuracy (%)')
plt.plot(validation_accuracy, label='Validation Accuracy (%)')
plt.title('Model Accuracy Over Epochs (RNN)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  
plt.legend()
plt.grid()
plt.show()

predictions = model.predict(X_test)

y_loc_test_rescaled = scaler_location.inverse_transform(y_loc_test)
y_mag_test_rescaled = scaler_magnitude.inverse_transform(y_mag_test)
pred_loc_rescaled = scaler_location.inverse_transform(predictions[:, :2])
pred_mag_rescaled = scaler_magnitude.inverse_transform(predictions[:, 2].reshape(-1, 1))

plt.figure(figsize=(12, 12))
scatter_actual = plt.scatter(
    y_loc_test_rescaled[:, 1], y_loc_test_rescaled[:, 0], 
    c=y_mag_test_rescaled.flatten(), cmap='Blues', label='Actual Data (Points)', marker='o', edgecolors='black'
)
plt.colorbar(scatter_actual, label='Actual Magnitude (Blues)')
scatter_predicted = plt.scatter(
    pred_loc_rescaled[:, 1], pred_loc_rescaled[:, 0], 
    c=pred_mag_rescaled.flatten(), cmap='Reds', label='Predicted Data (Crosses)', marker='x'
)
plt.colorbar(scatter_predicted, label='Predicted Magnitude (Reds)')
plt.title('Map of Predicted vs Actual Locations with Magnitudes (RNN)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid()
plt.show()

model.save('rnn_seismic_combined_model_colored.h5')
print("Model training complete. Model saved as 'rnn_seismic_combined_model_colored.h5'.")