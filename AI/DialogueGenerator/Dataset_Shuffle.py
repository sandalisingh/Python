import pandas as pd

# Load the dataset
file_path = "Conversation2.csv"
df = pd.read_csv(file_path)

# Remove the column "S.No."
df.drop(columns=['S.No.'], inplace=True)

# Shuffle the dataset
shuffled_df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled dataset, replacing the original file
shuffled_df.to_csv(file_path, index=False)

print("Dataset shuffled and saved successfully.")
