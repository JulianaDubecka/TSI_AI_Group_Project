import pandas as pd


def preprocess_data_to_dataframe(file_path):
    all_data = []

    with open(file_path, "r") as file:
        file.readline()  # Skip the first line
        for line in file:
            # Remove unwanted characters and format the string
            cleaned_line = (
                line.replace("np.float64", "")
                .replace("\n", "")
                .replace("(", "")
                .replace(")", "")
            )
            cleaned_line = cleaned_line.replace(", ", ",").replace(" ", ", ")

            # Ensure clean separation of numbers
            elements = cleaned_line.replace(" ", "").split(",")

            # Convert elements to float and append the list of floats to all_data
            try:
                float_list = [float(item) for item in elements]
                all_data.append(float_list)
            except ValueError as e:
                print(f"Error converting to float: {e} in line: {line}")

    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(
        all_data, columns=["epoch", "val_ndcg", "val_hr", "test_ndcg", "test_hr"]
    )

    df["epoch"] = df["epoch"].astype(int)

    for column in ["val_ndcg", "val_hr", "test_ndcg", "test_hr"]:
        df[column] = df[column].round(4)

    return df
