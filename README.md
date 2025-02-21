🎰 Ultimate AI-Powered Lottery Prediction System

📌 Overview
Welcome to the Ultimate AI-Powered Lottery Prediction System—an advanced, data-driven tool designed to enhance lottery predictions using artificial intelligence and statistical modeling. This project:

✅ Analyzes extensive historical lottery data to uncover trends and patterns.
✅ Utilizes cutting-edge AI & statistical models (Holt-Winters, ARIMA, GRU, Monte Carlo, XGBoost, Gradient Boosting, KNN) to generate precise number predictions.
✅ Identifies hot & cold number correlations to improve selection probability.
✅ Generates multiple predictions per draw, ensuring a balanced mix of frequently and infrequently drawn numbers.

⚙️ Installation
1️⃣ Clone the Repository
```bash
git clone https://github.com/KrisRz/Ultimate-AI-Powered-Lottery-Prediction-System-.git
cd lottery_provide
```
2️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

🚀 How to Run
To execute the full pipeline (data loading, analysis, and prediction), run:
```bash
python main.py
```
To generate only lottery predictions, run:
```bash
python scripts/predict_numbers.py
```

📂 Project Structure
```
lottery_provide/
│── data/                      # Folder for historical & latest lottery data
│   ├── lottery_results_final.csv    # Merged dataset used for predictions
│
│── scripts/                   # Core scripts for analysis & prediction
│   ├── analyze_data.py        # Analyzes historical trends
│   ├── predict_numbers.py     # Predicts the next draw
│
│── visualizations/            # Folder for prediction charts
│   ├── frequency_chart.png    # Visualized lottery number trends
│
│── models/                    # Folder for trained AI models
│   ├── gru_model.h5           # Saved GRU model
│   ├── arima_model.pkl        # Saved ARIMA model
│   ├── holt_winters_model.pkl # Saved Holt-Winters model
│   ├── xgboost_model.pkl      # Saved XGBoost model
│   ├── knn_model.pkl          # Saved K-Nearest Neighbors model
│   ├── gradient_boosting_model.pkl # Saved Gradient Boosting model
│   ├── linear_regression_model.pkl # Saved Linear Regression model
│
│── main.py                    # Master script to run all modules
│── requirements.txt            # List of dependencies
│── README.md                   # This file
```

🛠 Technologies Used
🔎 Data Processing: pandas, numpy
📈 Statistical Models: statsmodels (Holt-Winters, ARIMA)
🤖 Machine Learning Models: TensorFlow (GRU), XGBoost, KNN, Gradient Boosting, Linear Regression
📊 Data Visualization: matplotlib
🎰 Lottery Prediction Models: Monte Carlo Simulation, AI-Driven Number Selection

🔥 Features
✅ AI-Powered Predictions – Uses GRU, ARIMA, Holt-Winters, Monte Carlo, XGBoost, and Gradient Boosting for precision.
✅ Smart Number Selection – Optimized number coverage based on hot & cold frequencies.
✅ Historical Pattern Analysis – Identifies repeating trends from past draws.
✅ Optimized Processing – Handles large datasets with chunked data loading for efficiency.
✅ Multi-Model Integration – Merges multiple AI techniques for increased accuracy.

⚠️ Notes
🔹 This tool does not guarantee winning numbers (lotteries are inherently random). Use it for statistical insights and entertainment.
🔹 To update historical data, manually add new draw results to `data/lottery_results_final.csv`.

📜 License
🔖 MIT License – Open-source & free to use.

🚀 Ready to predict? Run `python main.py` and start analyzing! 🎰

