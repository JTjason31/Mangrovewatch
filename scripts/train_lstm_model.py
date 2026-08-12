import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X = np.load('data/lstm_X.npy')
y = np.load('data/lstm_y.npy')

# X already has shape (samples, timesteps, features=3) - no reshape needed
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 70/30 train-test split, preserving time order (no shuffle, since it's a time series)
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

model = keras.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2])),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(16),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTest MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

model.save('models/lstm_ndvi_forecaster.keras')
print("\nModel saved to models/lstm_ndvi_forecaster.keras")