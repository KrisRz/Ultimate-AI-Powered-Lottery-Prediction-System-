import os
from xgboost import XGBRegressor
import pandas as pd
import random
import numpy as np
from collections import Counter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import logging
import gc
from tqdm import tqdm  # For progress bars

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress unnecessary TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Path to the merged lottery results file
MERGED_FILE = os.path.join("data", "lottery_results_final.csv")

# Constants
MAX_OCCURRENCE = 2  # Maximum occurrences of a number in generated sets
LOOK_BACK = 5  # Look-back window for LSTM
LSTM_UNITS = 32  # Increased LSTM units for better learning
LSTM_EPOCHS = 5  # Increased epochs for better accuracy
BATCH_SIZE = 4  # Batch size for LSTM training
MONTE_CARLO_SIMULATIONS = 10000  # Number of Monte Carlo simulations

# --- Function to save and load models ---
def save_model_to_disk(model, model_name, folder="models"):
    """Saves a model to disk."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    if model_name.endswith(".h5"):
        model.save(os.path.join(folder, model_name))
    else:
        joblib.dump(model, os.path.join(folder, model_name))

def load_model_from_disk(model_name, folder="models"):
    """Loads a model from disk."""
    if not os.path.exists(folder):
        return None
    if model_name.endswith(".h5"):
        return load_model(os.path.join(folder, model_name))
    else:
        return joblib.load(os.path.join(folder, model_name))

# --- Function to train LSTM model ---
def train_lstm_model(series, look_back=LOOK_BACK):
    """Trains an LSTM model on historical lottery data."""
    if len(series) < look_back:
        logging.warning("⚠️ Not enough data for LSTM training.")
        return None, MinMaxScaler(), None  # Ensure scaler is always returned

    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))

    def generate_sequences(data, look_back):
        for i in range(len(data) - look_back):
            yield data[i:i + look_back], data[i + look_back]

    X, y = zip(*generate_sequences(series_scaled, look_back))
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    last_sequence = X[-1].reshape(1, look_back, 1)

    # Clear memory
    del X, y
    gc.collect()

    return model, scaler, last_sequence

# --- Function to select hot and cold numbers ---
def select_hot_and_cold_numbers(number_counts):
    """Selects hot and cold numbers based on frequency."""
    hot_numbers = list(number_counts.nlargest(15).index)
    cold_numbers = list(number_counts.nsmallest(15).index)

    if not hot_numbers:
        hot_numbers = random.sample(range(1, 60), 15)
    if not cold_numbers:
        cold_numbers = random.sample(range(1, 60), 15)

    return hot_numbers, cold_numbers

# --- AI Prediction Models ---
def predict_with_arima(number_counts):
    """Predicts numbers using the ARIMA model."""
    try:
        model = ARIMA(number_counts, order=(1, 1, 1)).fit()
        return [num for num in model.forecast(steps=5).astype(int) if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ ARIMA failed: {e}")
        return []

def predict_with_holt_winters(number_counts):
    """Predicts numbers using the Holt-Winters model."""
    try:
        model = ExponentialSmoothing(number_counts, trend="add", seasonal=None).fit()
        return [num for num in model.forecast(5).astype(int) if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ Holt-Winters failed: {e}")
        return []

def predict_with_linear_regression(number_counts):
    """Predicts numbers using Linear Regression."""
    try:
        X = np.arange(len(number_counts)).reshape(-1, 1)
        y = number_counts.values
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(np.arange(len(number_counts), len(number_counts) + 5).reshape(-1, 1))
        return [num for num in predictions.astype(int).flatten() if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ Linear Regression failed: {e}")
        return []

def predict_with_xgboost(number_counts):
    """Predicts numbers using XGBoost."""
    try:
        X = np.arange(len(number_counts)).reshape(-1, 1)
        y = number_counts.values
        model = XGBRegressor()
        model.fit(X, y)
        predictions = model.predict(np.arange(len(number_counts), len(number_counts) + 5).reshape(-1, 1))
        return [num for num in predictions.astype(int).flatten() if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ XGBoost failed: {e}")
        return []

def predict_with_knn(number_counts):
    """Predicts numbers using K-Nearest Neighbors."""
    try:
        X = np.arange(len(number_counts)).reshape(-1, 1)
        y = number_counts.values
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, y)
        predictions = model.predict(np.arange(len(number_counts), len(number_counts) + 5).reshape(-1, 1))
        return [num for num in predictions.astype(int).flatten() if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ KNN failed: {e}")
        return []

def predict_with_gradient_boosting(number_counts):
    """Predicts numbers using Gradient Boosting."""
    try:
        X = np.arange(len(number_counts)).reshape(-1, 1)
        y = number_counts.values
        model = GradientBoostingClassifier()
        model.fit(X, y)
        predictions = model.predict(np.arange(len(number_counts), len(number_counts) + 5).reshape(-1, 1))
        return [num for num in predictions.astype(int).flatten() if 1 <= num <= 59]
    except Exception as e:
        logging.warning(f"⚠️ Gradient Boosting failed: {e}")
        return []

# --- Monte Carlo Simulation ---
def monte_carlo_simulation(df, num_simulations=MONTE_CARLO_SIMULATIONS):
    """Performs Monte Carlo simulation to predict numbers."""
    all_numbers = df.values.flatten()
    number_counts = Counter(all_numbers)
    probabilities = {num: count / sum(number_counts.values()) for num, count in number_counts.items()}
    simulations = [random.choices(list(probabilities.keys()), weights=probabilities.values(), k=6) for _ in range(num_simulations)]
    best_numbers = Counter([num for sim in simulations for num in sim]).most_common(6)
    return [num[0] for num in best_numbers]

# --- Main Prediction Function ---
def analyze_and_predict():
    """Loads lottery data, trains models, and prepares number predictions."""
    if not os.path.exists(MERGED_FILE):
        logging.error("❌ Error: Merged data file not found!")
        return [], [], [], [], [], [], [], [], [], []

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

        # Ensure time index for time series models
        number_counts.index = pd.date_range(start="2000-01-01", periods=len(number_counts), freq="D")

        # Select hot and cold numbers
        hot_numbers, cold_numbers = select_hot_and_cold_numbers(number_counts)

        # AI predictions
        arima_forecast = predict_with_arima(number_counts)
        holt_forecast = predict_with_holt_winters(number_counts)
        linear_forecast = predict_with_linear_regression(number_counts)
        xgboost_forecast = predict_with_xgboost(number_counts)
        knn_forecast = predict_with_knn(number_counts)
        gradient_boosting_forecast = predict_with_gradient_boosting(number_counts)
        monte_carlo_forecast = monte_carlo_simulation(df[num_cols])

        # LSTM prediction
        series = np.array(number_counts.index)
        lstm_model = load_model_from_disk("lstm_model.h5")
        
        # Ensure the scaler is always initialized
        scaler = MinMaxScaler()

        if lstm_model is None:
            lstm_model, scaler, last_sequence = train_lstm_model(series)
            if lstm_model:  # Save only if model was successfully trained
                save_model_to_disk(lstm_model, "lstm_model.h5")

        lstm_forecast = (
            [num for num in scaler.inverse_transform(lstm_model.predict(last_sequence)).astype(int).flatten() if 1 <= num <= 59]
            if lstm_model and last_sequence is not None else []
        )

        # Ensure predictions are not empty
        if not lstm_forecast:
            logging.warning("⚠️ LSTM failed to generate predictions, filling with random hot numbers.")
            lstm_forecast = random.sample(hot_numbers, min(6, len(hot_numbers)))

        # Clear memory
        del df, all_numbers, number_counts, series
        gc.collect()

        return hot_numbers, cold_numbers, arima_forecast, holt_forecast, linear_forecast, xgboost_forecast, knn_forecast, gradient_boosting_forecast, monte_carlo_forecast, lstm_forecast

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        return [], [], [], [], [], [], [], [], [], []

        # Analyze numbers
        all_numbers = df[num_cols].values.flatten()
        number_counts = pd.Series(all_numbers).value_counts().sort_index()

        # Select hot and cold numbers
        hot_numbers, cold_numbers = select_hot_and_cold_numbers(number_counts)

        # AI predictions
        arima_forecast = predict_with_arima(number_counts)
        holt_forecast = predict_with_holt_winters(number_counts)
        linear_forecast = predict_with_linear_regression(number_counts)
        xgboost_forecast = predict_with_xgboost(number_counts)
        knn_forecast = predict_with_knn(number_counts)
        gradient_boosting_forecast = predict_with_gradient_boosting(number_counts)
        monte_carlo_forecast = monte_carlo_simulation(df[num_cols])

        # LSTM prediction
        series = np.array(number_counts.index)
        lstm_model = load_model_from_disk("lstm_model.h5")
        if lstm_model is None:
            lstm_model, scaler, last_sequence = train_lstm_model(series)
            save_model_to_disk(lstm_model, "lstm_model.h5")
        lstm_forecast = (
            [num for num in scaler.inverse_transform(lstm_model.predict(last_sequence)).astype(int).flatten() if 1 <= num <= 59]
            if lstm_model
            else []
        )

        # Clear memory
        del df, all_numbers, number_counts, series
        gc.collect()

        return hot_numbers, cold_numbers, arima_forecast, holt_forecast, linear_forecast, xgboost_forecast, knn_forecast, gradient_boosting_forecast, monte_carlo_forecast, lstm_forecast

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        return [], [], [], [], [], [], [], [], [], []

# --- Generate Predictions ---
def generate_multiple_predictions(n=5):
    """Generates multiple sets of lottery predictions."""
    predictions = []
    used_numbers = Counter()

    for i in tqdm(range(n), desc="Generating Predictions"):
        logging.info(f"🔄 Generating Prediction {i+1}")

        numbers = set()
        hot_numbers, cold_numbers, arima, holt, linear, xgboost, knn, gradient_boosting, monte, lstm = analyze_and_predict()

        # Combine all AI forecasts (filter None values)
        ai_forecasts = list(filter(None, arima + holt + linear + xgboost + knn + gradient_boosting + monte + lstm))
        random.shuffle(ai_forecasts)

        logging.info(f"🔍 AI Predictions Available: {ai_forecasts}")

        # Add numbers from AI forecasts (limit occurrence)
        for num in ai_forecasts:
            if num not in numbers and used_numbers[num] < MAX_OCCURRENCE:
                logging.info(f"✅ Adding {num} from AI predictions.")
                numbers.add(num)
                used_numbers[num] += 1
                if len(numbers) >= 6:
                    break

        # Fill remaining slots with hot/cold numbers
        available_numbers = list(set(hot_numbers + cold_numbers) - numbers)
        random.shuffle(available_numbers)

        while len(numbers) < 6 and available_numbers:
            num = available_numbers.pop()
            logging.info(f"⚠️ Filling with hot/cold number: {num}")
            numbers.add(num)
            used_numbers[num] += 1

        # If still not 6 numbers, pad with random numbers
        while len(numbers) < 6:
            num = random.randint(1, 59)
            if num not in numbers:
                logging.info(f"⚠️ Padding with random number: {num}")
                numbers.add(num)

        predictions.append(sorted(numbers))

    logging.info(f"✅ Final Predictions: {predictions}")
    return predictions

if __name__ == "__main__":
    predictions = generate_multiple_predictions(5)
    if predictions:
        logging.info("\n🔮 Final Predictions:")
        for i, pred in enumerate(predictions, 1):
            logging.info(f"🎯 Prediction {i}: {pred}")
    else:
        logging.warning("⚠️ No predictions generated.")
