import pandas as pd

# Define the path to your large dataset CSV file
large_dataset_path = "Datasets/Synthetic-Persona-Chat_train.csv"

# Define the number of smaller datasets you want to split into
num_splits = 10  # You can change this number as per your requirement

# Load the large dataset into a Pandas DataFrame
large_dataset = pd.read_csv(large_dataset_path)

# Calculate the number of rows in each smaller dataset
rows_per_split = len(large_dataset) // num_splits

# Split the large dataset into smaller datasets
for i in range(num_splits):
    start_index = i * rows_per_split
    end_index = (i + 1) * rows_per_split if i < num_splits - 1 else None
    
    # Extract the subset for this split
    smaller_dataset = large_dataset.iloc[start_index:end_index]
    
    # Define the path to save the smaller dataset CSV file
    smaller_dataset_path = f"Datasets/Synthetic-Persona-Chat_train_{i + 1}.csv"
    
    # Save the smaller dataset to a CSV file
    smaller_dataset.to_csv(smaller_dataset_path, index=False)
    
    print(f"Smaller dataset {i + 1} saved to {smaller_dataset_path}")
