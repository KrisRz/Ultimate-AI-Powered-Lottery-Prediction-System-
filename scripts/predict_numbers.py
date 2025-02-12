import os
import pandas as pd
import random
import numpy as np
from itertools import combinations
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress unnecessary TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Path to the lottery results file
MERGED_FILE = os.path.join("data", "lottery_results_final.csv")

# --- Train LSTM Model ---
def train_lstm_model(series, look_back=10):
    """Trains an LSTM model on historical lottery data with memory optimization."""
    if len(series) < look_back:
        logging.warning("⚠️ Not enough data for LSTM training.")
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))
    
    def generate_sequences(data, look_back):
        for i in range(len(data) - look_back):
            yield data[i:i + look_back], data[i + look_back]
    
    X, y = zip(*generate_sequences(series_scaled, look_back))
    X, y = np.array(X), np.array(y)
    
    # LSTM Model
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(32, return_sequences=False),  # Fewer units for optimization
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=1, verbose=0)  # Fewer epochs
    
    return model, scaler, X[-1].reshape(1, look_back, 1)

# --- Analyze and Predict ---
def analyze_and_predict():
    """Loads lottery data, trains models, and prepares number predictions."""
    if not os.path.exists(MERGED_FILE):
        logging.error("❌ Error: Merged data file not found!")
        return [], [], [], [], [], []

    try:
        # Load data in chunks for memory efficiency
        chunksize = 10**5
        df = pd.concat([chunk for chunk in pd.read_csv(MERGED_FILE, chunksize=chunksize)])
        num_cols = ["N1", "N2", "N3", "N4", "N5", "N6", "BN"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype("Int64")
        df.dropna(subset=num_cols, inplace=True)

        # Flatten numbers into a single list and count occurrences
        all_numbers = df[num_cols].values.flatten()
        number_counts = pd.Series(all_numbers).value_counts().sort_index()

        # Select hot and cold numbers
        hot_numbers = [num for num in number_counts.nlargest(15).index if 1 <= num <= 59]
        cold_numbers = [num for num in number_counts.nsmallest(15).index if 1 <= num <= 59]

        if not hot_numbers:
            logging.warning("⚠️ No hot numbers found. Using random fallback.")
            hot_numbers = random.sample(range(1, 60), 15)
        if not cold_numbers:
            logging.warning("⚠️ No cold numbers found. Using random fallback.")
            cold_numbers = random.sample(range(1, 60), 15)

        # Generate most common pairs and triplets
        def generate_pairs_and_triplets(data):
            if len(data) == 0:
                return []
            for row in data:
                yield from combinations(row, 2)
                yield from combinations(row, 3)
        
        pairs_and_triplets = list(generate_pairs_and_triplets(df[num_cols].values))
        pair_counts = pd.Series(pairs_and_triplets).value_counts().nlargest(10) if pairs_and_triplets else pd.Series([])

        if pair_counts.empty:
            logging.warning("⚠️ No frequent pairs found. Using random hot numbers as fallback.")
            pair_counts = pd.Series(random.sample(hot_numbers, 10))

        # --- ARIMA Forecasting ---
        try:
            arima_model = ARIMA(number_counts, order=(1, 1, 1))  # Simpler ARIMA model
            arima_fit = arima_model.fit()
            arima_forecast = pd.Series([num for num in arima_fit.forecast(steps=5).astype(int).tolist() if 1 <= num <= 59])
        except Exception as e:
            logging.warning(f"⚠️ ARIMA prediction failed: {e}. Using random fallback.")
            arima_forecast = pd.Series(random.sample(hot_numbers, min(5, len(hot_numbers))))

        # --- Holt-Winters Forecasting (fallback if ARIMA fails) ---
        if arima_forecast.empty:
            try:
                exp_smooth = ExponentialSmoothing(number_counts, trend="add", seasonal=None).fit()
                trending_numbers = pd.Series([num for num in exp_smooth.forecast(5).astype(int).tolist() if 1 <= num <= 59])
            except Exception as e:
                logging.warning(f"⚠️ Holt-Winters prediction failed: {e}. Using random fallback.")
                trending_numbers = pd.Series(random.sample(hot_numbers, min(5, len(hot_numbers))))
        else:
            trending_numbers = pd.Series([])

        # --- LSTM Prediction ---
        series = np.array(number_counts.index)
        lstm_model, scaler, last_sequence = train_lstm_model(series)

        if lstm_model:
            lstm_forecast = pd.Series([num for num in scaler.inverse_transform(lstm_model.predict(last_sequence)).astype(int).flatten().tolist() if 1 <= num <= 59])
        else:
            logging.warning("⚠️ LSTM prediction failed. Using random fallback.")
            lstm_forecast = pd.Series(random.sample(hot_numbers, min(5, len(hot_numbers))))

        return hot_numbers, cold_numbers, list(pair_counts), trending_numbers, arima_forecast, lstm_forecast

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        return [], [], [], [], [], []

# --- Generate Multiple Predictions ---
def generate_multiple_predictions(n=5):
    """Generates multiple sets of lottery predictions using a hybrid AI approach."""
    hot_numbers, cold_numbers, pair_counts, trending_numbers, arima_forecast, lstm_forecast = analyze_and_predict()

    if not hot_numbers or not cold_numbers:
        logging.error("❌ No numbers available for prediction!")
        return []

    predictions = []
    used_numbers = {}
    logging.info("\n🔮 Generating multiple predictions...")

    for i in range(n):
        predicted_numbers = set()
        
        if not arima_forecast.empty:
            predicted_numbers.add(random.choice(arima_forecast.tolist()))

        if not trending_numbers.empty:
            predicted_numbers.add(random.choice(trending_numbers.tolist()))

        if not lstm_forecast.empty:
            predicted_numbers.add(random.choice(lstm_forecast.tolist()))

        while len(predicted_numbers) < 6:
            num = random.choice(hot_numbers + cold_numbers)
            if used_numbers.get(num, 0) < 3:
                predicted_numbers.add(num)
                used_numbers[num] = used_numbers.get(num, 0) + 1
        
        final_numbers = sorted(predicted_numbers)
        predictions.append(final_numbers)
        logging.info(f"🎯 Prediction {i+1}: {final_numbers}")

    return predictions

if __name__ == "__main__":
    generate_multiple_predictions(5)
