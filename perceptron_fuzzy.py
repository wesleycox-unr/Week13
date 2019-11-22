# perceptron.py
# Written by: Wes Cox for IS485/485
# Oct 31, 2019
# Starting point for Question 1 of Programming Problem Set #7
#
# For a randomly generated set of data, perform the perceptron learning algorithm
# to correctly classify the points

import numpy
import matplotlib.pylab as plt
import pandas as pd
import sys
import copy

# Number of data points
N = 200

#    w0    w1  w2
initial_w = [0.1, 0.1, 0.1]

def getRand():
    return numpy.random.uniform(-1,1)

def sign(x1,x2,m,b):
    # If the x1 and x2 position lies above the line corresponding to m and b, then it is positive, otherwise negative
    liney = m*x1 + b
    
    percentWrong = 0.95


    if x2 > liney:
        if numpy.random.uniform(0,1) > percentWrong:
            return -1
        else:
            return 1
    else:
        if numpy.random.uniform(0,1) > percentWrong:
            return 1
        else:
            return -1


def createData(m, b):

    dataDict = {"x1": [], "x2": [], "yp":[]}

    # Create N datapoints
    for i in range(N):
        x1 = getRand()
        x2 = getRand()
        y = sign(x1,x2,m,b)

        dataDict['x1'].append(x1)
        dataDict['x2'].append(x2)
        dataDict['yp'].append(y)

    data = pd.DataFrame(dataDict)
    return data

def plotCurrentW(ww, currentData, flush = True):
    # Plot the randomly generated data
    plt.scatter(currentData[currentData["yp"] == 1]["x1"],currentData[currentData["yp"] == 1]["x2"], marker="o")
    plt.scatter(currentData[currentData["yp"] == -1]["x1"],currentData[currentData["yp"] == -1]["x2"], marker="x")
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Create data points corresponding the the weights so that they can be plotted on the graph
    wDict = {'x1':[], 'x2':[]}
    leftx2 = (ww[1] - ww[0])/ww[2]
    rightx2 = (-ww[1] - ww[0])/ww[2]

    wDict["x1"].append(-1.0)
    wDict["x2"].append(leftx2)
    wDict["x1"].append(1.0)
    wDict["x2"].append(rightx2)

    resultW = pd.DataFrame(wDict)

    # Plot the corresponding classification separating line
    plt.plot(resultW.x1, resultW.x2)


    if flush:
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    else:
        plt.show()
    

def perceptron(data):

    # Implement the perceptron learning algorithm to determine the weights w that will fit the data
    w = initial_w
    
    max_iterations = 100
    
    numMismatch = len(data) 
    bestW = w
   
    for iteration in range(0, max_iterations):
        print("Iteration {0} of {1} with w: {2}".format(iteration, max_iterations, w))
        
        miscategorized = []
        categorized = []
    
        for index in range(0, len(data)):
            row = data.iloc[index]
        
            h = w[0]*1 + w[1]*row["x1"] + w[2]*row["x2"]
    
            if numpy.sign(h) != numpy.sign(row["yp"]):
                # Mismatch
                miscategorized.append(index)
            else:
                categorized.append(index)
        
        print("Good: {0} and Bad: {1}".format(len(categorized),len(miscategorized)))
        
        
        if len(miscategorized) < numMismatch:
            # Better fit
            plotCurrentW(w, data)
            bestW = copy.deepcopy(w)
            numMismatch = len(miscategorized)
            
        if len(miscategorized) > 0:
            choice = miscategorized[numpy.random.randint(0, len(miscategorized))]
            w[0] = w[0] + data.iloc[choice]["yp"]*1
            w[1] = w[1] + data.iloc[choice]["yp"]*data.iloc[choice]["x1"]
            w[2] = w[2] + data.iloc[choice]["yp"]*data.iloc[choice]["x2"]
        else:
            print("Best w: {0} and with {1} mismatches".format(w,numMismatch)) 
            return w    
                
    # #print(data)
    
    # #for row in data.iterrows():
    # index = 0
    # while index < len(data):
        # print("Index {0} of {1}".format(index, len(data)))
        # plotCurrentW(w, data)
        # row = data.iloc[index]
        
        # #print(row)
        # #sys.exit()
        # #print(row[1]["x1"])
        # h = w[0]*1 + w[1]*row["x1"] + w[2]*row["x2"]
        
        # if numpy.sign(h) != numpy.sign(row["yp"]):
            # # Mismatch
            # # Update w values
            
            # #w[w0 w1 w2] = w + yp * data
            # w[0] = w[0] + row["yp"]*1
            # w[1] = w[1] + row["yp"]*row["x1"]
            # w[2] = w[2] + row["yp"]*row["x2"]
            # index = 0
        # else:
            # index += 1
    # # Steps:
    # # - See if the current weights will correctly predict the yp values in the DataFrame for all rows
    # # - If so, done.
    # # - If not, choose a row that isnt predicted correctly, and update the weights by the scalar product of yp with [1, x1, x2]
    # # - Repeat until all rows are correctly predicted

    print("Best w: {0} and with {1} mismatches".format(bestW,numMismatch)) 
    return bestW # Return the predicting weights

line_gradient = 1 # 45 degree line
intercept = 0 # passing through the origin
input_data = createData(line_gradient,intercept)

# Perform the learning
result_w = perceptron(input_data)

print(result_w)

flushPlot = False
plotCurrentW(result_w, input_data, flushPlot)