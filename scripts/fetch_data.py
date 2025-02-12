import os
import pandas as pd

# Define file paths
DATA_DIR = "data"
FILE_1 = os.path.join(DATA_DIR, "lottery_results_new1.csv")
FILE_2 = os.path.join(DATA_DIR, "lottery_results_new2.csv")
MERGED_FILE = os.path.join(DATA_DIR, "lottery_results_final.csv")

# Ensure directory exists
os.makedirs(DATA_DIR, exist_ok=True)

EXPECTED_COLUMNS = [
    "Draw_Number", "Day", "DD", "MMM", "YYYY",
    "N1", "N2", "N3", "N4", "N5", "N6", "BN",
    "Jackpot", "Wins", "Machine", "Set"
]

def clean_column_names(df):
    """Removes extra spaces from column names and standardizes them."""
    df.columns = df.columns.str.strip()  # Remove spaces
    return df

def validate_and_rename_columns(df):
    """Validates and renames columns to the expected format."""
    if len(df.columns) != len(EXPECTED_COLUMNS):
        print(f"❌ Column structure mismatch. Expected {len(EXPECTED_COLUMNS)} but found {len(df.columns)}")
        print(f"🔍 Detected columns: {df.columns.tolist()}")
        return None

    df.columns = EXPECTED_COLUMNS  # Rename columns
    return df

def standardize_date_format(df):
    """Converts DD, MMM, YYYY columns into a proper date format (YYYY-MM-DD)."""
    month_mapping = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
    }
    
    # Convert the day column to always have two digits
    df["DD"] = df["DD"].astype(str).str.zfill(2)  # ✅ Ensure DD has two digits

    # Convert month abbreviation to numeric
    df["Month_Num"] = df["MMM"].map(month_mapping)

    # Construct a proper date format
    df["Date"] = df["YYYY"].astype(str) + "-" + df["Month_Num"] + "-" + df["DD"]

    # Drop old columns
    df.drop(columns=["DD", "MMM", "YYYY", "Month_Num"], inplace=True)

    return df

def merge_lottery_data():
    """Loads, cleans, and merges two CSV files into one."""

    print("\n🔄 Merging lottery data from split CSV files...\n")

    try:
        # Load CSVs
        df1 = pd.read_csv(FILE_1)
        df2 = pd.read_csv(FILE_2)

        # Clean column names
        df1 = clean_column_names(df1)
        df2 = clean_column_names(df2)

        # Rename and validate columns
        df1 = validate_and_rename_columns(df1)
        df2 = validate_and_rename_columns(df2)

        if df1 is None or df2 is None:
            print("❌ Error: Column validation failed. Check the dataset structure.")
            return None

        # Standardize date format
        df1 = standardize_date_format(df1)
        df2 = standardize_date_format(df2)

        # Convert necessary columns to numeric
        numeric_cols = ["Draw_Number", "N1", "N2", "N3", "N4", "N5", "N6", "BN", "Jackpot", "Wins", "Set"]
        df1[numeric_cols] = df1[numeric_cols].apply(pd.to_numeric, errors='coerce').astype("Int64")
        df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors='coerce').astype("Int64")

        # Clean text columns
        df1["Machine"] = df1["Machine"].str.strip()
        df2["Machine"] = df2["Machine"].str.strip()
        df1["Day"] = df1["Day"].str.strip()
        df2["Day"] = df2["Day"].str.strip()

        # Merge datasets
        merged_df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)

        # Set date index for time-series analysis
        merged_df.set_index("Date", inplace=True)

        # Sort by draw number to maintain chronological order
        merged_df.sort_values(by="Draw_Number", inplace=True)

        # Save merged file
        merged_df.to_csv(MERGED_FILE)
        print(f"✅ Merged data saved to {MERGED_FILE}")

        return merged_df  # ✅ Return the merged DataFrame

    except Exception as e:
        print(f"❌ Error during merging: {e}")
        return None  # Return None if merging fails

if __name__ == "__main__":
    merge_lottery_data()
