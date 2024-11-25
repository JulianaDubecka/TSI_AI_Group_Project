from tqdm import tqdm
import pandas as pd

def preprocess_transactions(input_csv, output_txt, min_items=5, max_items=50, chunksize=100000):
    try:
        print("Reading the file in chunks...")
        session_data = []
        session_counter = 1

        # Read the file in chunks
        for chunk in tqdm(pd.read_csv(
            input_csv,
            sep=',',
            engine='python',
            encoding='utf-8',
            chunksize=chunksize,
            on_bad_lines='skip'
        ), desc="Processing chunks"):
            # Ensure t_dat is parsed as datetime
            chunk['t_dat'] = pd.to_datetime(chunk['t_dat'], errors='coerce')

            # Drop rows with invalid dates
            chunk = chunk.dropna(subset=['t_dat'])

            # Drop duplicate rows
            chunk = chunk.drop_duplicates()

            # Sort the chunk
            chunk = chunk.sort_values(by=['customer_id', 't_dat', 'article_id']).reset_index(drop=True)

            # Create session column
            chunk['session'] = (
                (chunk['customer_id'] != chunk['customer_id'].shift()) |
                (chunk['t_dat'] != chunk['t_dat'].shift())
            )
            chunk['session_id'] = chunk['session'].cumsum()

            # Filter and truncate sessions
            for session_id, group in chunk.groupby('session_id'):
                if min_items <= len(group) <= max_items:
                    # Add to session_data with truncated items
                    session_data.append(group.tail(max_items))
                    session_counter += 1

        # Combine all sessions
        print("Combining all valid sessions...")
        final_df = pd.concat(session_data, ignore_index=True)

        # Remap session IDs to sequential numbers
        print("Remapping session IDs...")
        session_id_mapping = {old_id: new_id for new_id, old_id in enumerate(final_df['session_id'].unique(), start=1)}
        final_df['session_id'] = final_df['session_id'].map(session_id_mapping)

        # Write to the output file
        print("Writing output to file...")
        with open(output_txt, 'w') as f:
            for session_id, group in tqdm(final_df.groupby('session_id'), desc="Writing sessions"):
                for article_id in group['article_id']:
                    f.write(f"{session_id} {article_id}\n")

        print(f"Processed data saved to {output_txt}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage (can be replaced with CLI or parameters)
if __name__ == "__main__":
    preprocess_transactions(
        input_csv="../data/raw/transactions_sample.csv",
        output_txt="../data/processed/h_m.txt",
        min_items=5,
        max_items=50
    )