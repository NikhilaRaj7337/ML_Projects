import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import time, tracemalloc

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# === Load and preprocess dataset ===
df = pd.read_csv(r'C:\Users\DELL\Documents\UGA files\Sem 2\Advanced Special Topics in CS (Wei Nui)\AST project\traffic.csv', index_col='DateTime', parse_dates=True)
df = df[df['Junction'] == 1]

# Feature engineering
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Normalize features
feature_cols = ['Vehicles', 'hour', 'dayofweek']
scalers = {col: MinMaxScaler() for col in feature_cols}
for col in feature_cols:
    df[[col]] = scalers[col].fit_transform(df[[col]])

# Sequence creation
def create_sequences(data, n_steps=12, target_index=0):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, target_index])
    return np.array(X), np.array(y)

data_array = df[feature_cols].values
X, y = create_sequences(data_array, n_steps=12)

# Train-test split
split_idx = len(X) - 96
train_X, test_X = X[:split_idx], X[split_idx:]
train_y, test_y = y[:split_idx], y[split_idx:]

# Build model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(train_X, train_y, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])

# Measure computational training time
train_start = time.time()
model.fit(train_X, train_y, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])
train_end = time.time()

# Evaluate with latency and memory
tracemalloc.start()
start_time = time.time()
predicted = model.predict(test_X, verbose=0)
end_time = time.time()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Metrics
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred).reshape(-1)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape_score = mape(test_y, predicted)
accuracy = 100 - mape_score

# Congestion level
predicted_unscaled = scalers['Vehicles'].inverse_transform(predicted)
congestion_level = (np.mean(predicted_unscaled) / 3000) * 100  # Assuming 3000 max/hour

print("=== Traffic Prediction Performance ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape_score:.2f}%")
print(f"Estimated Accuracy: {accuracy:.2f}%")
print(f"Congestion Level (Avg Predicted vs Max 3000): {congestion_level:.2f}%")
print(f"Latency (Prediction Time): {end_time - start_time:.4f} seconds")
print(f"Peak Memory Usage: {peak_mem / (1024 * 1024):.2f} MB")
print(f"Current Memory Usage: {current_mem / (1024 * 1024):.2f} MB")
print(f"Computational Training Time: {train_end - train_start:.4f} seconds")

# === Confusion Matrix ===
true_unscaled = scalers['Vehicles'].inverse_transform(test_y.reshape(-1, 1))
q1 = np.percentile(true_unscaled, 33)
q2 = np.percentile(true_unscaled, 66)

def get_label(val):
    if val < q1:
        return 'Low'
    elif val < q2:
        return 'Medium'
    else:
        return 'High'

true_labels = [get_label(x) for x in true_unscaled.flatten()]
predicted_labels = [get_label(x) for x in predicted_unscaled.flatten()]

cm = confusion_matrix(true_labels, predicted_labels, labels=['Low', 'Medium', 'High'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(6, 4))
disp.plot(cmap='Blues')
plt.title("Traffic Congestion Prediction - Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# === Dynamic Traffic Signal Operator Visualization ===
def traffic_light_status(avg_vehicles):
    if avg_vehicles < q1:
        return 'Green'
    elif avg_vehicles < q2:
        return 'Yellow'
    else:
        return 'Red'

signal = traffic_light_status(np.mean(predicted_unscaled))

color_map = {'Green': 'green', 'Yellow': 'orange', 'Red': 'red'}
plt.figure(figsize=(2, 6))
plt.title(f"Signal: {signal}")
plt.gca().set_facecolor('black')
plt.axis('off')

for i, light in enumerate(['Red', 'Yellow', 'Green']):
    plt.scatter(0.5, 0.8 - i * 0.3, s=3000, 
                color=color_map[light] if light == signal else 'gray')

plt.savefig("traffic_signal_status.png")
plt.show()