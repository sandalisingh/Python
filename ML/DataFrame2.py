# python3 DataFrame2.py

import pandas as pd

# 1. Acquiring dataset from GitHub Repository & creating dataframe 

url = 'https://raw.githubusercontent.com/sandalisingh/Iris-Data-Set/HOME.html/iris.csv'
dataFrame = pd.read_csv( url, header=None )     

# 2. Viewing dataframe [using head() and tail()]

print( '\n\n-> First 5 rows\n' )
print( dataFrame.head() )   # 5 by def

print( '\n\n-> Last 5 rows\n' )
print( dataFrame.tail() )

# 3. Print index range and columns

print( '\n\n-> Index Range & Columns\n' )
print( dataFrame.axes ) # returns [index range, columns, dtype]

# 4. Print dataframe information

print( '\n\n-> Dataframe Information\n' )
print( dataFrame.info() )

# 5. Print statistical summary of dataframe display

print( '\n\n-> Statistical Summary of Dataframe\n' )
print( dataFrame.describe() )

# 6. Print dataframe shape, number of rows, columns and elements

print( '\n\n-> Dataframe Shape = ' + str(dataFrame.shape) ) # index, cols
print( '-> Number of Rows = ' + str(len(dataFrame.index)) )
print( '-> Number of Columns = ' + str(len(dataFrame.columns)) + '\n' )

# 7. Drops columns of a dataset

dataFrame.drop( columns=[ 0, 3], axis=1, inplace=True )     
print( '\n\n-> After Dropping Columns 0 and 3\n')
print( print(dataFrame.head()) )

# 8. Print data using .loc and .iloc

print( '\n\n-> Data of Index 2 to 5(included)\n' )
print( dataFrame.loc[ 2:5 ] )  # retrieve rows  

# printing data using .iloc
print( '\n\n->Data of Index 2 to 5(not included) & Cols 1 to 4(not included)\n\n' )
print( dataFrame.iloc[ 2:5, 1:4] )   # retrieve rows

# print('\n\n\n\n')