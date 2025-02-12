import pandas as pd
import os
from itertools import combinations
from collections import Counter

# 📌 Define file path
MERGED_FILE = os.path.join("data", "lottery_results_final.csv")

EXPECTED_COLUMNS = [
    "Draw_Number", "Day", "Date", "N1", "N2", "N3", "N4", "N5", "N6", "BN",
    "Jackpot", "Wins", "Machine", "Set"
]

def analyze_lottery_data():
    """Analyzes historical lottery data with Wheel System and combination analysis."""

    if not os.path.exists(MERGED_FILE):
        print(f"❌ Error: {MERGED_FILE} not found! Run fetch_data.py first.")
        return

    try:
        # 📌 Load data
        df = pd.read_csv(MERGED_FILE)

        # 🔎 Debug: Check actual column names in the file
        print("\n📊 Actual Column Names in File:", df.columns.tolist())

        # 📌 Fix column name mismatches dynamically
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            df.columns = EXPECTED_COLUMNS
            print("✅ Fixed Column Names Automatically!")

        # 📌 Convert numeric columns to integers
        num_cols = ["N1", "N2", "N3", "N4", "N5", "N6", "BN"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

        # 📌 Drop missing values
        df.dropna(subset=num_cols, inplace=True)

        # 📌 Ensure final conversion to int
        df[num_cols] = df[num_cols].astype(int)

        # 🔎 Debug: Print column types
        print("\n🔎 Column types after conversion:\n", df[num_cols].dtypes)

        # 📌 Flatten all numbers into a single list
        all_numbers = df[num_cols].values.flatten()

        # 📌 Convert to a proper numeric Series
        number_counts = pd.Series(all_numbers).astype(int).value_counts().sort_index()

        # 📌 Most common number occurrences
        print("\n📈 Most Drawn Numbers (Top 10):")
        print(number_counts.nlargest(10))

        # 📌 Frequent number pairs & triplets
        pair_counts = Counter(tuple(sorted(pair)) for row in df[num_cols].values for pair in combinations(row, 2))
        triplet_counts = Counter(tuple(sorted(triplet)) for row in df[num_cols].values for triplet in combinations(row, 3))

        pair_counts = pd.Series(pair_counts).astype(int).nlargest(10)
        triplet_counts = pd.Series(triplet_counts).astype(int).nlargest(10)

        print("\n🔗 Most Common Pairs:")
        print(pair_counts)

        print("\n🔗 Most Common Triplets:")
        print(triplet_counts)

        # 📌 Implementacja Wheel System (analiza pokrycia liczbowego)
        wheel_coverage = {}
        for size in [6, 7, 8]:
            best_combinations = Counter(tuple(sorted(comb)) for draw in df[num_cols].values for comb in combinations(draw, size))
            wheel_coverage[size] = pd.Series(best_combinations).astype(int).nlargest(10)

        print("\n🎡 Wheel System Coverage (Most Frequent Combinations):")
        for size, combos in wheel_coverage.items():
            print(f"\n🔹 Best {size}-number sets:")
            print(combos)

    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

if __name__ == "__main__":
    analyze_lottery_data()
