import pandas as pd
import numpy as np
from collections import Counter

# List of emotions
EmotionList = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"]

def preprocessing_dataset(data):
    # Remove unnecessary columns
    columns_to_remove = ['id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear']
    data = data.drop(columns=columns_to_remove)
    print("-> Dropped extra columns")
    
    # Convert multi-label to single-label
    data['emotion'] = data[EmotionList].idxmax(axis=1)
    print("-> Converted multi label into single label target")

    data = data.drop(columns=EmotionList)
    print("-> Dropped multi labels")

    return data

def balance_dataset(df, target_col, remove_percentage=0.6):
    # Count occurrences of each label
    label_counts = df[target_col].value_counts()

    # Calculate the number of rows to remove for the "neutral" label
    neutral_count = label_counts.get('neutral', 0)
    remove_count = int(neutral_count * remove_percentage)

    # Randomly sample rows to remove
    if remove_count > 0:
        neutral_indices = df[df[target_col] == 'neutral'].index
        remove_indices = np.random.choice(neutral_indices, remove_count, replace=False)
        df_balanced = df.drop(remove_indices)
    else:
        df_balanced = df.copy()

    return df_balanced
    # Count the occurrences of each label
    label_counts = df['emotion'].value_counts()
    
    # Find the minority class
    minority_class = label_counts.idxmin()
    
    # Determine the number of samples in the minority class
    minority_count = label_counts.min()
    
    # Downsample the majority class
    majority_class = label_counts.idxmax()
    print("Majority Class = " + majority_class)

    majority_indices = df[df['emotion'] == majority_class].index
    downsampled_indices = df[df['emotion'] == minority_class].index
    downsampled_indices = downsampled_indices.append(
        majority_indices[:len(downsampled_indices) - minority_count]
    )
    balanced_df = df.loc[downsampled_indices]
    
    return balanced_df