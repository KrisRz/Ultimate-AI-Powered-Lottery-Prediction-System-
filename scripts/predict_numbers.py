import os
import logging  # ✅ Move this to the top
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # ✅ Add this import
from statsmodels.tsa.arima.model import ARIMA  # ✅ Add this import
from sklearn.ensemble import GradientBoostingClassifier  # ✅ Add this import
from xgboost import XGBClassifier  # ✅ Add this import
from sklearn.neighbors import KNeighborsClassifier  # ✅ Add this import
import joblib  # ✅ Add this import
import gc  # ✅ Add this import
import numpy as np
import pandas as pd
import random
from collections import Counter  # ✅ Add this import
import argparse  # ✅ Add this import
from tqdm import tqdm  # ✅ Add this import
from sklearn.linear_model import LinearRegression  # ✅ Add this import





# 🔹 **Disable GPU & Ensure Proper Execution Mode**
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode
tf.compat.v1.disable_eager_execution()  # Proper TF2 execution

# 🔹 **Setup Logging Correctly**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 **Define GRU Model**
def build_gru_model():
    model = Sequential([
        Input(shape=(10, 1)),
        GRU(32, return_sequences=False, activation="tanh"),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 🔹 **Run Training & Save Model**
if __name__ == "__main__":
    X = np.random.rand(100, 10, 1)  # Dummy data, replace with real dataset
    y = np.random.rand(100, 1)

    model = build_gru_model()

    model.fit(X, y, epochs=10, batch_size=4, verbose=1, use_multiprocessing=True)

    # Save trained model
    model.save("models/gru_model.h5")




# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress unnecessary TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Path to the merged lottery results file
MERGED_FILE = os.path.join("data", "lottery_results_final.csv")

# Constants
LOOK_BACK = 10  # Adjusted to match GRU input shape
LSTM_UNITS = 32  # Increased GRU units for better learning
LSTM_EPOCHS = 10  # Increased epochs for better accuracy
BATCH_SIZE = 4  # Batch size for GRU training
MONTE_CARLO_SIMULATIONS = 5000  # Reduced for faster processing


# Declare it as global at the beginning (outside functions)
used_numbers = None

def initialize_used_numbers():
    """Initialize the global used_numbers Counter."""
    global used_numbers
    if used_numbers is None:
        used_numbers = Counter()


# --- Function to save and load models ---
def save_model_to_disk(model, model_name, folder="models"):
    """Saves a model to disk."""
    os.makedirs(folder, exist_ok=True)
    if isinstance(model, (Sequential, tf.keras.Model)):
        model.save(os.path.join(folder, model_name))
    else:
        joblib.dump(model, os.path.join(folder, model_name))

def load_model_from_disk(model_name, folder="models"):
    """Loads a model from disk if available."""
    model_path = os.path.join(folder, model_name)
    if os.path.exists(model_path):
        if model_name.endswith(".h5"):
            return load_model(model_path)
        return joblib.load(model_path)
    return None

# --- Function to train GRU model ---
def train_gru_model(series, look_back=LOOK_BACK):
    """Trains a GRU model on historical lottery data."""
    if len(series) < look_back:
        logging.warning("⚠️ Not enough data for GRU training.")
        return None, MinMaxScaler(), None

    scaler = MinMaxScaler(feature_range=(0, 1))
    series_reshaped = series.astype(float).reshape(-1, 1)  # Ensure `series` has numeric values
    scaler.fit(series_reshaped)  # Fit the scaler
    series_scaled = scaler.transform(series_reshaped)  # Apply scaling

    def generate_sequences(data, look_back):
        for i in range(len(data) - look_back):
            yield data[i:i + look_back], data[i + look_back]

    X, y = zip(*generate_sequences(series_scaled, look_back))
    X, y = np.array(X), np.array(y)

    # Reshape X to (samples, LOOK_BACK, 1)
    X = X.reshape(-1, look_back, 1)

    model = Sequential([
        Input(shape=(look_back, 1)),
        GRU(LSTM_UNITS, return_sequences=False, activation="tanh"),  # Faster on Mac than LSTM
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    last_sequence = series_scaled[-look_back:].reshape(1, look_back, 1)
    logging.info(f"🔍 GRU Model Debugging:")
    logging.info(f"🔢 series.shape: {series.shape}")
    logging.info(f"🔢 series_scaled.shape: {series_scaled.shape}")
    logging.info(f"🔢 last_sequence.shape: {last_sequence.shape}")

    return model, scaler, last_sequence

# --- Function to select hot and cold numbers ---
# ✅ Ensure `used_numbers` is declared at the top level
used_numbers = Counter()

def select_hot_and_cold_numbers(number_counts):
    """Selects hot and cold numbers based on frequency, with a penalty for overuse."""
    global used_numbers  # ✅ Declare `global` inside the function where it is modified

    weights = np.linspace(1, 0.5, len(number_counts))  # Newest draws get higher weight
    weighted_counts = number_counts * weights

    hot_numbers = list(weighted_counts.nlargest(15).index)
    cold_numbers = list(weighted_counts.nsmallest(15).index)

    if not hot_numbers:
        hot_numbers = random.sample(range(1, 60), 15)
    if not cold_numbers:
        cold_numbers = random.sample(range(1, 60), 15)

    # ✅ Gradually reduce probability instead of removing numbers completely
    hot_numbers = [num for num in hot_numbers if random.random() > (0.1 * used_numbers[num])]
    cold_numbers = [num for num in cold_numbers if random.random() > (0.2 * used_numbers[num])]

    return hot_numbers, cold_numbers

# --- Unified AI Model Save/Load Function ---
def predict_with_model(number_counts, model_class, model_name, model_args=None):
    """Generic function to load, train, and save AI models automatically."""
    model_path = os.path.join("models", model_name)

    try:
        # Load the model if it exists
        if os.path.exists(model_path):
            logging.info(f"📂 Loading saved model: {model_name}")
            model = joblib.load(model_path)
        else:
            # Train new model if no saved version exists
            logging.info(f"⚙️ Training new model: {model_class.__name__}")
            X = np.arange(len(number_counts)).reshape(-1, 1)
            y = number_counts.values
            model = model_class(**(model_args or {}))
            model.fit(X, y)

            # Save the trained model
            joblib.dump(model, model_path)
            logging.info(f"✅ Model saved: {model_name}")

        # Predict next numbers
        predictions = model.predict(np.arange(len(number_counts), len(number_counts) + 5).reshape(-1, 1))
        return [num for num in predictions.astype(int).flatten() if 1 <= num <= 59]
    
    except Exception as e:
        logging.warning(f"⚠️ {model_class.__name__} failed: {e}")
        return []

# --- AI Prediction Models ---
def predict_with_linear_regression(number_counts):
    """Predicts numbers using Linear Regression."""
    return predict_with_model(number_counts, LinearRegression, "linear_regression_model.pkl")

def predict_with_xgboost(number_counts):
    """Predicts numbers using XGBoost."""
    return predict_with_model(number_counts, XGBRegressor, "xgboost_model.pkl")

def predict_with_knn(number_counts):
    """Predicts numbers using K-Nearest Neighbors."""
    return predict_with_model(number_counts, KNeighborsClassifier, "knn_model.pkl", {"n_neighbors": 5})

def predict_with_gradient_boosting(number_counts):
    """Predicts numbers using Gradient Boosting."""
    return predict_with_model(number_counts, GradientBoostingClassifier, "gradient_boosting_model.pkl")

def predict_with_arima(number_counts):
    """Predicts numbers using the ARIMA model with saving/loading."""
    model_path = os.path.join("models", "arima_model.pkl")
    
    try:
        # Load ARIMA model if saved
        if os.path.exists(model_path):
            logging.info("📂 Loading ARIMA model...")
            model = joblib.load(model_path)
        else:
            # Train new ARIMA model
            logging.info("⚙️ Training new ARIMA model...")
            model = ARIMA(number_counts, order=(3, 1, 2)).fit()
            joblib.dump(model, model_path)
            logging.info("✅ ARIMA model saved.")

        # Forecast next 5 numbers
        predictions = model.forecast(steps=5).astype(int)
        return [max(1, min(num, 59)) for num in predictions]
    
    except Exception as e:
        logging.warning(f"⚠️ ARIMA failed: {e}")
        return []

def predict_with_holt_winters(number_counts):
    """Predicts numbers using the Holt-Winters model, with saving/loading."""
    model_path = os.path.join("models", "holt_winters_model.pkl")

    try:
        # Load Holt-Winters model if saved
        if os.path.exists(model_path):
            logging.info("📂 Loading Holt-Winters model...")
            model = joblib.load(model_path)
        else:
            # Train new Holt-Winters model
            logging.info("⚙️ Training new Holt-Winters model...")
            model = ExponentialSmoothing(number_counts, trend="add", seasonal=None).fit()
            joblib.dump(model, model_path)
            logging.info("✅ Holt-Winters model saved.")

        # Forecast next 5 numbers
        predictions = model.forecast(steps=5).astype(int)
        return [max(1, min(num, 59)) for num in predictions]
    
    except Exception as e:
        logging.warning(f"⚠️ Holt-Winters failed: {e}")
        return []

# --- Monte Carlo Simulation ---
def monte_carlo_simulation(df, num_simulations=MONTE_CARLO_SIMULATIONS, decay_factor=0.90):
    """Performs Monte Carlo simulation with improved probability decay."""
    all_numbers = df.values.flatten()
    number_counts = Counter(all_numbers)

    # Apply probability decay
    probabilities = {num: (count / sum(number_counts.values())) ** decay_factor for num, count in number_counts.items()}

    # Normalize probabilities
    total_prob = sum(probabilities.values())
    probabilities = {num: prob / total_prob for num, prob in probabilities.items()}

    simulations = [random.choices(list(probabilities.keys()), weights=probabilities.values(), k=6) for _ in range(num_simulations)]
    best_numbers = Counter([num for sim in simulations for num in sim]).most_common(6)

    return [num[0] for num in best_numbers]

# --- Main Prediction Function ---
def analyze_and_predict(fast_mode=False):
    """Loads lottery data, trains models, and prepares number predictions."""
    
    if not os.path.exists(MERGED_FILE):
        logging.error("❌ Error: Merged data file not found!")
        return [], [], [], [], [], [], [], [], [], []

    try:
        # ✅ Load dataset with validation
        df = load_lottery_data(MERGED_FILE)
        if df is None or df.empty:
            logging.error("❌ Error: Loaded dataset is empty!")
            return [], [], [], [], [], [], [], [], [], []
        
        num_cols = ["N1", "N2", "N3", "N4", "N5", "N6", "BN"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype("Int64")
        df.dropna(subset=num_cols, inplace=True)

        if df.empty:
            logging.error("❌ Error: Dataset is empty after cleaning!")
            return [], [], [], [], [], [], [], [], [], []

        # ✅ Extract & analyze number frequencies
        all_numbers = df[num_cols].values.flatten()
        number_counts = pd.Series(all_numbers).value_counts().sort_index()

        # Ensure time index for time series models
        number_counts.index = range(len(number_counts))

        # ✅ Select hot and cold numbers
        hot_numbers, cold_numbers = select_hot_and_cold_numbers(number_counts)

        # ✅ AI Predictions with Safe Default Values
        if not fast_mode:
            arima_forecast = predict_with_arima(number_counts) or []
            holt_forecast = predict_with_holt_winters(number_counts) or []
            linear_forecast = predict_with_linear_regression(number_counts) or []
            xgboost_forecast = predict_with_xgboost(number_counts) or []
            knn_forecast = predict_with_knn(number_counts) or []
            gradient_boosting_forecast = predict_with_gradient_boosting(number_counts) or []
            monte_carlo_forecast = monte_carlo_simulation(df[num_cols]) or []
            gru_forecast = predict_with_gru(number_counts) or []

        # ✅ Memory Cleanup
        del df, all_numbers, number_counts
        gc.collect()

        # ✅ Fast Mode: Return only essential values
        if fast_mode:
            return hot_numbers, cold_numbers, [], [], [], [], [], [], monte_carlo_forecast, []

        # ✅ Ensure all returned values are lists to avoid unpacking errors
        return (
            hot_numbers, cold_numbers, arima_forecast, holt_forecast,
            linear_forecast, xgboost_forecast, knn_forecast,
            gradient_boosting_forecast, monte_carlo_forecast, gru_forecast
        )

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}", exc_info=True)
        return [], [], [], [], [], [], [], [], [], []


        
        

    # Constants
LOOK_BACK = 10  # Adjusted to match GRU input shape
LSTM_UNITS = 32  # Increased GRU units for better learning
LSTM_EPOCHS = 10  # Increased epochs for better accuracy
BATCH_SIZE = 4  # Batch size for GRU training
MONTE_CARLO_SIMULATIONS = 5000  # Reduced for faster processing
MODEL_PATH = "models/gru_model.h5"  # ✅ FIXED: Defined here





        # Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def predict_with_gru(number_counts):
    """Predicts multiple lottery numbers using a trained GRU model."""

    # Convert index to numeric array
    series = np.array(number_counts.index, dtype=np.float32)

    # Ensure enough data
    if len(series) < LOOK_BACK:
        logging.warning(f"⚠️ Not enough data for GRU training. Required: {LOOK_BACK}, Available: {len(series)}")
        return random.sample(range(1, 60), 6)  # Fallback to random numbers

    # Load the saved GRU model
    if os.path.exists(MODEL_PATH):
        logging.info("📂 Loading saved GRU model...")
        gru_model = load_model(MODEL_PATH)
    else:
        logging.warning("⚠️ No saved GRU model found. Returning random numbers.")
        return random.sample(range(1, 60), 6)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_reshaped = series.reshape(-1, 1)
    series_scaled = scaler.fit_transform(series_reshaped)

    # Debugging
    logging.info("🔍 GRU Model Debugging:")
    logging.info(f"🔢 Raw series shape: {series.shape}")
    logging.info(f"🔢 Scaled series shape: {series_scaled.shape}")

    # Get the last sequence for prediction
    last_sequence = series_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)

    # Debugging: Check shape before prediction
    logging.info(f"🔢 Last sequence shape before prediction: {last_sequence.shape}")

    try:
        # Predict next 6 lottery numbers
        predicted_values = gru_model.predict(last_sequence)

        # Debugging: Check raw predicted values
        logging.info(f"🔢 Raw predicted values (before inverse transform): {predicted_values}")

        # Ensure inverse transform is possible
        if predicted_values.ndim == 2:
            # **Fix Scaling: Apply Inverse Transform Correctly**
            predicted_values = scaler.inverse_transform(predicted_values)

            # **Flatten the output and ensure numbers are within range**
            predicted_values = predicted_values.flatten()
            logging.info(f"🔢 Scaled predicted values (after inverse transform): {predicted_values}")

            gru_forecast = []
            seen = set()

            for num in predicted_values.flatten():
                num = max(1, min(int(num), 59))  # Ensure valid range
                if num not in seen:
                    gru_forecast.append(num)
                    seen.add(num)
                if len(gru_forecast) == 6:
                    break

            # If GRU still outputs less than 6 numbers, generate extra until we have 6
            while len(gru_forecast) < 6:
                num = random.randint(1, 59)
                if num not in seen:
                    gru_forecast.append(num)
                    seen.add(num)

        else:
            logging.warning("⚠️ GRU output shape mismatch, using random numbers.")
            gru_forecast = random.sample(range(1, 60), 6)

    except Exception as e:
        logging.warning(f"⚠️ GRU prediction failed: {e}")
        gru_forecast = random.sample(range(1, 60), 6)

    # ✅ Correctly indented return statement inside the function
    logging.info(f"🎯 Final GRU Forecast: {gru_forecast}")
    return gru_forecast  # ✅ This must be inside the function


def generate_multiple_predictions(n=5, fast_mode=False):
    """Generates multiple sets of lottery predictions."""
    predictions = []
    initialize_used_numbers()  # Ensure `used_numbers` is initialized


    for i in tqdm(range(n), desc="Generating Predictions"):
        logging.info(f"🔄 Generating Prediction {i+1}")

        # Get AI-generated numbers
        hot_numbers, cold_numbers, arima, holt, linear, xgboost, knn, gradient_boosting, monte, gru = analyze_and_predict(fast_mode)

        # Balanced AI weighting to prevent bias from one model
        ai_forecasts = (
            random.sample(arima, min(2, len(arima))) +
            random.sample(holt, min(2, len(holt))) +
            random.sample(linear, min(2, len(linear))) +
            random.sample(xgboost, min(2, len(xgboost))) +
            random.sample(knn, min(2, len(knn))) +
            random.sample(gradient_boosting, min(2, len(gradient_boosting))) +
            random.sample(monte, min(3, len(monte))) +  # Monte Carlo gets a slight boost
            random.sample(gru, min(2, len(gru)))
        )

        ai_forecasts = list(set(ai_forecasts))  # Ensure no duplicates
        ai_forecasts = [num for num in ai_forecasts if 1 <= num <= 59]  # Ensure valid range

        # Penalize numbers based on how frequently they appear
        weighted_forecasts = [(num, max(1, 5 - used_numbers[num])) for num in ai_forecasts]  # Lower weight if used more
        weighted_forecasts = sorted(weighted_forecasts, key=lambda x: x[1], reverse=True)  # Sort by weight

        # Select numbers based on weighted penalties
        ai_forecasts = [num[0] for num in weighted_forecasts[:15]]  # Take top 15 based on weight

        # **Ensure Proper Indentation Here**
        numbers = set(random.sample(ai_forecasts, min(3, len(ai_forecasts))))  # ✅ Fixes indentation

        while len(numbers) < 6:
            num = random.randint(1, 59)
            if num not in numbers and used_numbers[num] < 2:
                numbers.add(num)

        predictions.append(sorted(numbers))

        # Update used numbers count
        for num in numbers:
            used_numbers[num] += 1

    return predictions


# --- Optimized Data Loading ---
def load_lottery_data(filepath, max_rows=10_000):
    """Loads only the last `max_rows` rows from the dataset efficiently."""
    total_rows = sum(1 for _ in open(filepath)) - 1  # Get total row count
    skip_rows = max(0, total_rows - max_rows)  # Rows to skip
    df = pd.read_csv(filepath, skiprows=range(1, skip_rows), usecols=["N1", "N2", "N3", "N4", "N5", "N6", "BN"], dtype="Int64")
    return df

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick prediction mode")
    args = parser.parse_args()

    predictions = generate_multiple_predictions(5, fast_mode=args.fast)

    if predictions:
        logging.info("\n🔮 Final Predictions:")
        for i, pred in enumerate(predictions, 1):
            logging.info(f"🎯 Prediction {i}: {pred}")
