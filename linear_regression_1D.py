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


#    w0    w1  w2
initial_w = [0.0, 0.0, 0.0]

def plotCurrentW(ww, currentData, delay, newPoint = None):
    # Plot the randomly generated data
    plt.scatter(currentData["x"],currentData["yp"], marker=".")
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("x")
    plt.ylabel("y")

    if newPoint:
        y = ww[0] + ww[1]*newPoint
        plt.scatter(newPoint,y, marker="x", c="r")

    # Create data points corresponding the the weights so that they can be plotted on the graph
    wDict = {'x':[], 'yp':[]}
    leftx2 = ww[0] - ww[1]
    rightx2 = ww[0] + ww[1]

    wDict["x"].append(-1.0)
    wDict["yp"].append(leftx2)
    wDict["x"].append(1.0)
    wDict["yp"].append(rightx2)

    resultW = pd.DataFrame(wDict)

    # Plot the corresponding classification separating line
    plt.plot(resultW.x, resultW.yp)


    plt.draw()
    plt.pause(delay)
    plt.clf()
    
def getEin(ww, data):
    Ein = 0
    for index in range(0,len(data)):
        h = ww[0] + ww[1]*data["x"][index]
        Ein += (h - data["yp"][index])*(h - data["yp"][index])

    Ein = Ein/len(data)

    return Ein
    
    
def regression(data):

    # Implement the perceptron learning algorithm to determine the weights w that will fit the data
    w = copy.deepcopy(initial_w)
    
    
    max_iterations = 300
    iteration = 0
    
    learning_rate = 0.5
    
    bestW = copy.deepcopy(initial_w)
    minEin = getEin(initial_w, data)
    
    while iteration < max_iterations:
        index = numpy.random.randint(0, len(data))
        
        print("Iteration: {0}, index {1}, w {2}".format(iteration, index, w))
        
        if iteration%10 == 0:
            plotCurrentW(w, data, 0.001)
        
        h = w[0] + w[1]*data["x"][index]
        
        
        Ein = getEin(w, data)
        if Ein < minEin:
            minEin = Ein
            bestW = copy.deepcopy(w)
        
        x = [1, data['x'][index]]
        
        derivative = [2*elem*(h - data["yp"][index])/len(data) for elem in x]
        
        w[0] = w[0] - learning_rate*derivative[0]
        w[1] = w[1] - learning_rate*derivative[1]
        
        
        
        
        
        iteration += 1
    
    
    return bestW # Return the predicting weights

input_data = pd.read_json("linear_data_1D.txt")
print(input_data)

# Perform the learning
result_w = regression(input_data)

print(result_w)

initialEin = getEin(initial_w, input_data)
bestEin = getEin(result_w, input_data)

print("Initial Ein {0} result Ein {1}".format(initialEin, bestEin))

plotCurrentW(result_w, input_data, 5, 0.3)

