import time
from scripts import fetch_data, analyze_data, predict_numbers

def main():
    """Main function to load, analyze, and predict lottery numbers."""
    
    print("\n🔄 Loading lottery data...")
    start_time = time.time()
    
    # Merge CSV files and return the merged DataFrame
    df = fetch_data.merge_lottery_data()
    
    if df is None or df.empty:
        print("❌ Merging failed or no data found. Exiting program.")
        return
    
    print(f"✅ Data Loaded in {time.time() - start_time:.2f} seconds.\n")

    print("\n📊 Analyzing historical lottery data...")
    try:
        start_time = time.time()
        analyze_data.analyze_lottery_data()
        print(f"✅ Analysis Completed in {time.time() - start_time:.2f} seconds.\n")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return

    print("\n🔮 Generating multiple predictions using a hybrid model...")
    try:
        start_time = time.time()
        predictions = predict_numbers.generate_multiple_predictions(5)

        print(f"\n✅ Prediction Completed in {time.time() - start_time:.2f} seconds.\n")

        # 🔥 PODSUMOWANIE NA KOŃCU 🔥
        print("\n🎯🎯 FINAL PREDICTED NUMBERS FOR THE NEXT DRAW 🎯🎯")
        for idx, numbers in enumerate(predictions, 1):
            print(f"🔹 Set {idx}: {numbers}")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

if __name__ == "__main__":
    main()
