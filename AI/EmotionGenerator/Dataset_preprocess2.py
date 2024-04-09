import csv
import pandas as pd

# Assuming your CSV file is named 'dataset.csv', change this to the appropriate path if necessary
csv_file_path = 'Dataset/new.csv'
df = pd.DataFrame(columns=['chat_text', 'text_response'])

# Open the CSV file
with open(csv_file_path, 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Loop over each row in the CSV file
    for row in csv_reader:
        print(row)  
        lines = row[0].strip().split('\n')
        print(lines)

        user1 = ''
        user2 = ''
        
        for line in lines:
            print(line)
            if line.startswith("User 1:"):
                user1 += line.split('User 1: ')[1] 
            elif line.startswith("User 2:"):
                user2 = line.split('User 2: ')[1]

                if user1 != '':
                    df = df._append({'chat_text': user1, 'text_response': user2}, ignore_index=True)
                    user1 = ''
                    user2 = ''

    df.to_csv('Dataset/new_2.csv', index=False)

