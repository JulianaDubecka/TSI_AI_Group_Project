
import pandas as pd

def debug_dataset(input_file, output_file):
    try:
        print("Loading dataset for debugging...")
        # Load dataset into DataFrame
        df = pd.read_csv(input_file, sep=" ", header=None, names=["session_id", "article_id"])
        
        # Inspect first few rows
        print("First 10 rows of the dataset:")
        print(df.head(10))

        # Check for missing values
        print("\nMissing values in each column:")
        print(df.isnull().sum())

        # Check data types
        print("\nColumn data types:")
        print(df.dtypes)

        # Validate numeric conversion
        print("\nEnsuring session_id and article_id are integers...")
        df["session_id"] = pd.to_numeric(df["session_id"], errors="coerce")
        df["article_id"] = pd.to_numeric(df["article_id"], errors="coerce")
        
        # Identify invalid rows
        invalid_rows = df[df.isnull().any(axis=1)]
        print(f"\nNumber of invalid rows: {len(invalid_rows)}")
        if not invalid_rows.empty:
            print("Sample of invalid rows:")
            print(invalid_rows.head())

        # Drop invalid rows
        print("\nDropping invalid rows...")
        df = df.dropna().astype({"session_id": int, "article_id": int})
        print(f"Remaining rows after cleaning: {len(df)}")

        # Save the cleaned dataset
        print(f"Saving cleaned dataset to {output_file}...")
        df.to_csv(output_file, sep=" ", index=False, header=False)
        print("Dataset cleaned and saved successfully.")

    except Exception as e:
        print(f"An error occurred during debugging: {e}")

if __name__ == "__main__":
    input_file = "../data/processed/h_m.txt"  # Update with the path to your dataset
    output_file = "../sasrec/data/h_m_cleaned.txt"  # Path for the cleaned dataset
    debug_dataset(input_file, output_file)
