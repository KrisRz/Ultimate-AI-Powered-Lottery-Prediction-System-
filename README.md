# 🎰 Lottery Prediction System

## 📌 Overview
This project is a **UK Lottery Prediction System** that:

✅ Fetches the latest **UK National Lottery** results.  
✅ Analyzes **historical lottery data (2016-present)**.  
✅ Uses **AI & statistical models** (*Holt-Winters, ARIMA, LSTM*) to predict the next draw.  
✅ Identifies **correlations** between frequently drawn numbers.  
✅ Displays final **predictions with visualizations**.  

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/KrisRz/Ultimate-AI-Powered-Lottery-Prediction-System-.git
cd lottery_provide
```

### 2️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run
To execute the **full pipeline** (fetch, analyze, predict), run:
```bash
python main.py
```

To generate **only lottery predictions**, run:
```bash
python scripts/predict_numbers.py
```

---

## 📂 Project Structure
```
lottery_provide/
│── data/                      # Folder for historical & latest lottery data
│   ├── lottery_results_new1.csv    # Main dataset (2016-present)
│   ├── lottery_results_new2.csv
│── scripts/                   # Core scripts for analysis & prediction
│   ├── fetch_data.py          # Fetches latest 180 days of results
│   ├── analyze_data.py        # Analyzes historical trends
│   ├── predict_numbers.py     # Predicts the next draw
│
│── visualizations/            # Folder for prediction charts
│   ├── frequency_chart.png    # Visualized lottery number trends
│
│── models/                    # Folder for trained AI models
│   ├── lstm_model.h5          # Saved LSTM model
│   ├── arima_model.pkl        # Saved ARIMA model
│   ├── holt_winters_model.pkl # Saved Holt-Winters model
│   ├── rl_model.zip           # Saved Reinforcement Learning model
│
│── main.py                    # Master script to run all modules
│── requirements.txt           # List of dependencies
│── README.md                  # This file
```

---

## 🛠 Technologies Used

- **🔎 Data Scraping:** `BeautifulSoup`, `requests`
- **📊 Data Analysis:** `pandas`, `numpy`
- **📈 Statistical Models:** `statsmodels` (*Holt-Winters, ARIMA*)
- **🤖 Machine Learning:** `TensorFlow` (*LSTM for time series forecasting*)
- **📉 Visualization:** `matplotlib`

---

## 🔥 Features

✅ **Automated Data Fetching** – Always uses the latest lottery results.  
✅ **Multiple AI & Statistical Models** – Combines **LSTM, ARIMA, Holt-Winters** for accuracy.  
✅ **Historical Trend Analysis** – Identifies **frequently drawn numbers** over time.  
✅ **Correlation Analysis** – Detects **commonly paired & triplet numbers**.  
✅ **Smart Number Selection** – Uses **hot & cold numbers + AI models** for best predictions.  
✅ **Interactive Visualizations** – Shows **frequency distributions of past draws**.  

---

## ⚠️ Notes

🔹 **This tool does not guarantee winning numbers** (*lotteries are random*). Use it for **statistical insights only**.  
🔹 To **update historical data beyond 180 days**, manually **merge a larger dataset**.

---

## 📜 License

🔖 **MIT License** – Open-source & free to use.  

---

🚀 **Ready to try?** Run `python main.py` and start analyzing! 🎰

