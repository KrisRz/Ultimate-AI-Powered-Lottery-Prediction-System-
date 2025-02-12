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

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ukrywamy zbędne logi TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ścieżka do pliku z wynikami loterii
MERGED_FILE = os.path.join("data", "lottery_results_final.csv")

# --- 📌 Funkcja trenowania LSTM (optymalizowana pod kątem pamięci) ---
def train_lstm_model(series, look_back=10):
    """Trains an LSTM model on historical lottery data with memory optimization."""
    if len(series) < look_back:
        logging.warning("⚠️ Not enough data for LSTM training.")
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))
    
    # Używamy generatora zamiast listy, aby zaoszczędzić pamięć
    def generate_sequences(data, look_back):
        for i in range(len(data) - look_back):
            yield data[i:i + look_back], data[i + look_back]
    
    X, y = zip(*generate_sequences(series_scaled, look_back))
    X, y = np.array(X), np.array(y)
    
    # Prostszy model LSTM z mniejszą liczbą jednostek
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(32, return_sequences=False),  # Zmniejszona liczba jednostek
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=1, verbose=0)  # Mniej epok
    
    return model, scaler, X[-1].reshape(1, look_back, 1)

# --- 📌 Główna funkcja predykcji (optymalizowana pod kątem pamięci) ---
def analyze_and_predict():
    """Loads lottery data, trains models, and prepares number predictions with memory optimization."""
    if not os.path.exists(MERGED_FILE):
        logging.error("❌ Error: Merged data file not found!")
        return []

    try:
        # --- 📌 Ładowanie danych w partiach ---
        chunksize = 10**5  # Przetwarzanie danych w partiach
        df = pd.concat([chunk for chunk in pd.read_csv(MERGED_FILE, chunksize=chunksize)])
        num_cols = ["N1", "N2", "N3", "N4", "N5", "N6", "BN"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype("Int64")
        df.dropna(subset=num_cols, inplace=True)

        # --- 📌 Analiza liczb ---
        all_numbers = df[num_cols].values.flatten()
        number_counts = pd.Series(all_numbers).value_counts().sort_index()

        # --- 📌 Wybór liczb gorących i zimnych ---
        hot_numbers = [num for num in number_counts.nlargest(15).index if 1 <= num <= 59]
        cold_numbers = [num for num in number_counts.nsmallest(15).index if 1 <= num <= 59]

        # Obsługa pustych list
        if not hot_numbers:
            logging.warning("⚠️ No hot numbers found. Using random numbers as fallback.")
            hot_numbers = random.sample(range(1, 60), 15)  # Losowe liczby jako zapas
        if not cold_numbers:
            logging.warning("⚠️ No cold numbers found. Using random numbers as fallback.")
            cold_numbers = random.sample(range(1, 60), 15)  # Losowe liczby jako zapas

        # --- 📌 Analiza najczęstszych par i trójek (używamy generatorów) ---
        def generate_pairs_and_triplets(data):
            if len(data) == 0:  # Sprawdzenie, czy dane istnieją
                return []
            for row in data:
                yield from combinations(row, 2)
                yield from combinations(row, 3)
        
        pairs_and_triplets = list(generate_pairs_and_triplets(df[num_cols].values))
        pair_counts = pd.Series(pairs_and_triplets).value_counts().nlargest(10) if pairs_and_triplets else []

        # Obsługa pustej listy pair_counts
        if not pair_counts:
            logging.warning("⚠️ No frequent pairs found. Using random hot numbers as fallback.")
            pair_counts = random.sample(hot_numbers, 10)  # Losowe liczby z gorących numerów

        # --- 📌 Predykcja ARIMA ---
        try:
            arima_model = ARIMA(number_counts, order=(1, 1, 1))  # Prostszy model ARIMA
            arima_fit = arima_model.fit()
            arima_forecast = [num for num in arima_fit.forecast(steps=5).astype(int).tolist() if 1 <= num <= 59]
        except Exception as e:
            logging.warning(f"⚠️ ARIMA prediction failed: {e}. Using random hot numbers as fallback.")
            arima_forecast = random.sample(hot_numbers, min(5, len(hot_numbers)))  # Awaryjne numery

        # --- 📌 Predykcja Holt-Winters (tylko jeśli ARIMA zawodzi) ---
        if not arima_forecast:
            try:
                exp_smooth = ExponentialSmoothing(number_counts, trend="add", seasonal=None).fit()
                trending_numbers = [num for num in exp_smooth.forecast(5).astype(int).tolist() if 1 <= num <= 59]
            except Exception as e:
                logging.warning(f"⚠️ Holt-Winters prediction failed: {e}. Using random hot numbers as fallback.")
                trending_numbers = random.sample(hot_numbers, min(5, len(hot_numbers)))  # Awaryjne numery
        else:
            trending_numbers = []

        # --- 📌 Predykcja LSTM ---
        series = np.array(number_counts.index)
        lstm_model, scaler, last_sequence = train_lstm_model(series)

        if lstm_model:
            lstm_forecast = [num for num in scaler.inverse_transform(lstm_model.predict(last_sequence)).astype(int).flatten().tolist() if 1 <= num <= 59]
        else:
            logging.warning("⚠️ LSTM prediction failed. Using random hot numbers as fallback.")
            lstm_forecast = random.sample(hot_numbers, min(5, len(hot_numbers)))  # Awaryjne numery

        return hot_numbers, cold_numbers, list(pair_counts), trending_numbers, arima_forecast, lstm_forecast

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        return []

# --- 📌 Generowanie wielu zestawów (optymalizowane pod kątem pamięci) ---
def generate_multiple_predictions(n=5):
    """Generates multiple sets of lottery predictions using a hybrid method with memory optimization."""
    hot_numbers, cold_numbers, pair_counts, trending_numbers, arima_forecast, lstm_forecast = analyze_and_predict()

    if not hot_numbers or not cold_numbers:
        logging.error("❌ No numbers available for prediction!")
        return []

    predictions = []
    used_numbers = {}
    logging.info("\n🔮 Generating multiple predictions...")

    for i in range(n):
        predicted_numbers = set()
        
        # Dodajemy 1 liczbę z ARIMA
        if arima_forecast:
            num = random.choice(arima_forecast)
            predicted_numbers.add(num)
            used_numbers[num] = used_numbers.get(num, 0) + 1
        
        # Dodajemy 1 liczbę z Holt-Winters
        if trending_numbers:
            num = random.choice(trending_numbers)
            predicted_numbers.add(num)
            used_numbers[num] = used_numbers.get(num, 0) + 1
        
        # Dodajemy 1 liczbę z LSTM
        if lstm_forecast:
            num = random.choice(lstm_forecast)
            predicted_numbers.add(num)
            used_numbers[num] = used_numbers.get(num, 0) + 1
        
        # Dodajemy 3 liczby z hot/cold numbers
        while len(predicted_numbers) < 6:
            num = random.choice(hot_numbers + cold_numbers)
            if used_numbers.get(num, 0) < 3:  # Limit dla gorących/zimnych liczb
                predicted_numbers.add(num)
                used_numbers[num] = used_numbers.get(num, 0) + 1
        
        final_numbers = sorted(predicted_numbers)
        predictions.append(final_numbers)
        logging.info(f"🎯 Prediction {i+1}: {final_numbers}")

    return predictions

if __name__ == "__main__":
    generate_multiple_predictions(5)