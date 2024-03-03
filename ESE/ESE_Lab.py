# python3 ESE_Lab.py

''' IMPORT LIBRARIES  '''

import numpy as numpy
from matplotlib import pyplot as pyplot
import pandas as pandas
from IPython.display import display
from sklearn import metrics
from DecisionTree import *
import pprint   # Data pretty printer

''' IMPORT DS '''

url = 'SCP1.csv'
DS = pandas.read_csv( url )

print("\n\n-> DATASET SHAPE\n",DS.shape)
print( '\n\n-> FIRST 5 ROWS\n' )
print(DS.head(5))
print("\n\n")

''' DESCRIPTION OF DATASET '''
print( '\n\n-> DATASET DESCRIPTION  \n' )
print( DS.describe() )

''' EXTRACTING DEPENDENT & INDEPENDENT VARIABLES  '''

X = DS.iloc[:, :-1].values
Y = DS.iloc[:, -1].values

''' DATA SIMULATION '''
x = numpy.arange(0,50)
x = pandas.DataFrame(({'x':x}))

y1 = numpy.random.uniform(10,15,10)
y2 = numpy.random.uniform(20,25,10)
y3 = numpy.random.uniform(0,5,10)
y4 = numpy.random.uniform(30,32,10)
y5 = numpy.random.uniform(13,17,10)

y = numpy.concatenate((y1,y2,y3,y4,y5))
y = y[:,None]

''' SCATTER PLOT '''
print( '\n\n-> XY SHAPE  \n' )
print( x.shape, y.shape )

print( '\n\n-> SCATTER PLOT OF DATA\n' )
pyplot.figure(figsize=(7,5))
pyplot.plot(x,y, 'o')
pyplot.title("Scatter plot of x vs. y")
pyplot.xlabel("x")
pyplot.ylabel("y")
pyplot.show()

''' GRADIENT BOOSTING '''
''' DECISION TREE IN A LOOP '''

xi = x # initialization of input
yi = y # initialization of target
# x,y --> use where no need to change original y
ErrorResiduals = 0 # initialization of error
NoOfRows = len(yi)  # number of rows
Predicted_ouptut = 0 # initial prediction 0

for i in range(30): # like n_estimators
    print( '\n\n-> ITERATION ' + str(i+1) + ' \n' )

    TREE = DecisionTree(xi,yi)

    # print( 'DECISION TREE ' + str(i) )
    # pprint.pprint( TREE ) 

    TREE.find_better_split(0)
    
    r = numpy.where(xi == TREE.split)[0][0]    
    
    left_idx = numpy.where(xi <= TREE.split)[0]
    right_idx = numpy.where(xi > TREE.split)[0]
    
    Prediction_Col = numpy.zeros(NoOfRows)   # gives a new array of the given shape filled with zeroes
    numpy.put(Prediction_Col, left_idx, numpy.repeat(numpy.mean(yi[left_idx]), r))  # replace left side mean y
    numpy.put(Prediction_Col, right_idx, numpy.repeat(numpy.mean(yi[right_idx]), NoOfRows-r))  # right side mean y
    
    Prediction_Col = Prediction_Col[:,None]   # make long vector (nx1) in compatible with y
    Predicted_ouptut = Predicted_ouptut + Prediction_Col  # final prediction will be previous prediction value + new prediction of residual
    
    ErrorResiduals = y - Predicted_ouptut  # needed originlly here as residual always from original y   

    # Model Accuracy, how often is the classifier correct?
    print("SCORE : ",metrics.explained_variance_score(y, Predicted_ouptut)) 
    print("MEAN ABSOLUTE ERROR : ",metrics.mean_absolute_error(y, Predicted_ouptut)) 
    print("MEAN SQUARED ERROR : ",metrics.mean_squared_error(y, Predicted_ouptut)) 
    print("R2 SCORE : ",metrics.r2_score(y, Predicted_ouptut)) 
    
    ''' FEED ERROR RESIDUAL TO NEXT DECISION TREE '''
    yi = ErrorResiduals # update yi as residual to reloop
    
    ''' PLOTTING GRAPHS '''
    xa = numpy.array(x.x) # column name of x is x 
    order = numpy.argsort(xa)
    xs = numpy.array(xa)[order]
    ys = numpy.array(Predicted_ouptut)[order]
    
    #epreds = np.array(epred[:,None])[order]

    f, (PLOT1, PLOT2) = pyplot.subplots(1, 2, sharey=True, figsize = (13,2.5))

    ''' COMPARING DATA POINTS WITH THE PREDICTION OF THE DECISION TREE '''
    PLOT1.plot(x,y, 'o')  # plot actual data points
    PLOT1.plot(xs, ys, 'r')   # plot predicted output
    PLOT1.set_title(f'Prediction ( Iteration - {i+1} )')
    PLOT1.set_xlabel('x')
    PLOT1.set_ylabel('actual_y / predicted_y')

    ''' PLOTTING ERROR RESIDUALS CORRESPONDING EACH X '''
    PLOT2.plot(x, ErrorResiduals, 'go')
    PLOT2.set_title(f'ErrorResiduals vs x ( Iteration - {i+1} )')
    PLOT2.set_xlabel('x')
    PLOT2.set_ylabel('ErrorResiduals')

    pyplot.show()




    