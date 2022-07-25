# python3 DataFrame3.py

#Importing the required libraries
import pandas as pd
from matplotlib import pyplot as plt

#Importing and reading a csv dataset using Pandas
url = 'https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv'
dataFrame = pd.read_csv( url )
print(dataFrame.head(10))
print("\n\n")

#Outlier detection
for x in dataFrame.index:
    if dataFrame.loc[x,"Attack"] > 80:
        dataFrame.loc[x,"Attack"] = 120
print(dataFrame.head(10))
print("\n\n")

#Handling missing values
empty_remove = dataFrame.dropna()
print(empty_remove.head(10))
print("\n\n")

dataFrame["Type 2"].fillna("Grass", inplace = True)
print(dataFrame.head(10))
print("\n\n")

#analysing redundancy
dataFrame.drop_duplicates(inplace = True)
print(dataFrame.head(10))
print("\n\n")

#normalizing two columns - Attack and Defense
cols_to_norm = ['Attack','Defense']
dataFrame[cols_to_norm] = dataFrame[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(dataFrame.head(10))
print("\n\n")
