# python3 DecisionTreeID3.py

# DECISION TREE UD3 (Iterative(repeatedly) Dichotomiser(divides) 3)
import numpy as np  # fast maths fxns, read & manipulate numpy arrays
import pandas as pd
from numpy import log2 as log
import pprint   # Data pretty printer

eps = np.finfo(float).eps   # machine epsilon
dataFrame = pd.read_csv( 'PlayTennis.csv' )

print( "\nDATAFRAME\n" )
print( dataFrame )

def entropyOfDS( DF ):
    Label = DF.keys()[-1]   # last column    # target variable vector
    TotalEntropy = 0     # initialising entropy  # finds the entropy of that class
    ClassList = DF[Label].unique()  # unique values from last column of the dataframe
    for Class in ClassList:
        p = DF[Label].value_counts()[Class] / DF.shape[0] 
        # (no of times 'yes'/'no' occurs)/(total rows)
        TotalEntropy -= ( p * np.log2(p) )  # Entropy = -Summation(p*log(p))
    return TotalEntropy
  
# to calculate entropy of each attribute
def getEntropyAttribute( DF, attr ):
  Label = DF.keys()[-1]   # play tennis
  entropy2 = 0
  for variable in DF[attr].unique():    # for each unique value in "attribute" col
      entropy = 0
      for target_variable in DF[Label].unique():  # for each unique value in last col # yes & no
          Numerator = len( DF[attr] [DF[attr]==variable] [DF[Label]==target_variable] )
          Denominator = len( DF[attr] [DF[attr]==variable] )    # no of 
          fxn = Numerator / ( Denominator + eps )
          entropy += - fxn * log( fxn + eps )
      fraction2 = Denominator / len( DF )
      entropy2 += -fraction2*entropy
  return abs(entropy2)  # absolute value


def getWinner(DF):
    Entropy_att = []
    IG = [] # Info Gain
    for key in DF.keys()[:-1]:   # for each col excluding the last
        # IG(attr) = entropy of dataset - entropy of attribute
        IG.append( entropyOfDS(DF) - getEntropyAttribute(DF,key) )
    return DF.keys()[:-1][np.argmax(IG)] # returns attr with max IG
  
  
def getSubtable(DF, node, value):
  return DF[DF[node] == value].reset_index(drop=True)


def createDecisionTree(DF, Tree = None): 
    Class = DF.keys()[-1]   # last col
    
    node = getWinner(DF) # max IG attr
    
    distinctValues = np.unique(DF[node])
    
    # empty dictionary to create tree    
    if Tree is None:                    
        Tree = {}
        Tree[node] = {}
    
    for value in distinctValues:
        
        subtable = getSubtable(DF, node, value)
        clValue,counts = np.unique( subtable['PlayTennis'], return_counts = True )                        
        
        if len(counts) == 1:  # stops if subset is pure
            Tree[node][value] = clValue[0]                                                    
        else:        
            Tree[node][value] = createDecisionTree(subtable) # Calling the function recursively 
                   
    return Tree

print( "\nDESCION TREE\n" )
pprint.pprint( createDecisionTree(dataFrame) )    # Prints the formatted representation of object on stream

