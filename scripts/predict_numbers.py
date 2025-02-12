import os
import pandas as pd
import random
import numpy as np
from itertools import combinations
from collections import Counter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress unnecessary TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Path to the merged lottery results file
MERGED_FILE = os.path.join("data", "lottery_results_final.csv")

# Maximum times a number can appear across all predictions
MAX_OCCURRENCE = 2
LOOK_BACK = 5  # Reduced look-back window for LSTM
LSTM_EPOCHS = 2  # Fewer epochs for faster execution
LSTM_UNITS = 16  # Fewer LSTM units for reduced complexity

# --- Function to train LSTM model ---
def train_lstm_model(series):
    """Trains an LSTM model on historical lottery data."""
    if len(series) < LOOK_BACK:
        logging.warning("⚠️ Not enough data for LSTM training.")
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))

    def generate_sequences(data, look_back):
        for i in range(len(data) - look_back):
            yield data[i:i + look_back], data[i + look_back]

    X, y = zip(*generate_sequences(series_scaled, LOOK_BACK))
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(LOOK_BACK, 1)),
        LSTM(LSTM_UNITS, return_sequences=False),  # Fewer LSTM units
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=2, verbose=0)

    # Store the last sequence before deleting X and y
    last_sequence = X[-1].reshape(1, LOOK_BACK, 1)

    # Clear memory
    del X, y
    gc.collect()

    return model, scaler, last_sequence

# --- Function to select hot and cold numbers dynamically ---
def select_hot_and_cold_numbers(number_counts):
    """Selects hot and cold numbers dynamically."""
    hot_numbers = [num for num in number_counts.nlargest(15).index if 1 <= num <= 59]
    cold_numbers = [num for num in number_counts.nsmallest(15).index if 1 <= num <= 59]

    if not hot_numbers:
        hot_numbers = random.sample(range(1, 60), 15)
    if not cold_numbers:
        cold_numbers = random.sample(range(1, 60), 15)

    return hot_numbers, cold_numbers

# --- AI Predictions ---
def predict_with_arima(number_counts):
    """Predicts numbers using the ARIMA model."""
    try:
        if len(number_counts) < 10:
            return []
        arima_model = ARIMA(number_counts, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        return [num for num in arima_fit.forecast(steps=5).astype(int).tolist() if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ ARIMA failed: {e}")
        return []

def predict_with_holt_winters(number_counts):
    """Predicts numbers using the Holt-Winters model."""
    try:
        exp_smooth = ExponentialSmoothing(number_counts, trend="add", seasonal=None).fit()
        return [num for num in exp_smooth.forecast(5).astype(int).tolist() if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ Holt-Winters failed: {e}")
        return []

# --- Main Prediction Function ---
def analyze_and_predict():
    """Loads lottery data, trains models, and prepares number predictions."""
    if not os.path.exists(MERGED_FILE):
        logging.error("❌ Error: Merged data file not found!")
        return [], [], [], [], []

    try:
        # Load data in chunks to save memory
        chunksize = 10**5  # Adjust based on your dataset size
        df = pd.concat([chunk for chunk in pd.read_csv(MERGED_FILE, chunksize=chunksize)])
        num_cols = ["N1", "N2", "N3", "N4", "N5", "N6", "BN"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype("Int64")
        df.dropna(subset=num_cols, inplace=True)

        # Analyze numbers
        all_numbers = df[num_cols].values.flatten()
        number_counts = pd.Series(all_numbers).value_counts().sort_index()

        # Select hot and cold numbers
        hot_numbers, cold_numbers = select_hot_and_cold_numbers(number_counts)

        # AI predictions
        arima_forecast = predict_with_arima(number_counts)
        trending_numbers = predict_with_holt_winters(number_counts)

        # LSTM prediction
        series = np.array(number_counts.index)
        lstm_model, scaler, last_sequence = train_lstm_model(series)
        lstm_forecast = (
            [num for num in scaler.inverse_transform(lstm_model.predict(last_sequence)).astype(int).flatten().tolist() if 1 <= num <= 59]
            if lstm_model
            else []
        )

        # Clear memory
        del df, all_numbers, number_counts, series
        gc.collect()

        return hot_numbers, cold_numbers, trending_numbers, arima_forecast, lstm_forecast

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        return [], [], [], [], []

# --- Generate Predictions ---
def generate_multiple_predictions(n=5):
    """Generates multiple sets of lottery predictions."""
    hot_numbers, cold_numbers, trending_numbers, arima_forecast, lstm_forecast = analyze_and_predict()
    if not hot_numbers or not cold_numbers:
        logging.error("❌ No numbers available for prediction!")
        return []

    predictions = []
    used_numbers = Counter()

    for _ in range(n):
        predicted_numbers = set()

        # Ensure fair AI model contribution
        ai_forecasts = [arima_forecast, trending_numbers, lstm_forecast]
        for forecast in ai_forecasts:
            random.shuffle(forecast)
            for num in forecast:
                if num not in predicted_numbers and used_numbers[num] < MAX_OCCURRENCE:
                    predicted_numbers.add(num)
                    used_numbers[num] += 1
                    break

        # Fill remaining slots with hot/cold numbers ensuring no duplication
        available_numbers = list(set(hot_numbers + cold_numbers) - predicted_numbers)
        random.shuffle(available_numbers)

        while len(predicted_numbers) < 6 and available_numbers:
            num = available_numbers.pop()
            if num not in predicted_numbers and used_numbers[num] < MAX_OCCURRENCE:
                predicted_numbers.add(num)
                used_numbers[num] += 1

        predictions.append(sorted(predicted_numbers))

    return predictions

if __name__ == "__main__":
    predictions = generate_multiple_predictions(5)
    if predictions:
        logging.info("\n🔮 Final Predictions:")
        for i, pred in enumerate(predictions, 1):
            logging.info(f"🎯 Prediction {i}: {pred}")
    else:
        logging.warning("⚠️ No predictions generated.")
