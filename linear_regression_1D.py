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

    # Plot the predicted datapoint
    if newPoint:
        y = ww[0] + ww[1]*newPoint
        plt.scatter(newPoint,y, marker="x", c="r")

    # Create data points corresponding the the weights so that they can be plotted on the graph
    wDict = {'x':[], 'yp':[]}
    
    # y = yIntercept + gradient*x
    #   = w[0] + w[1]*x
    # To plot a straight line, we need the x and y at the far left (x=-1) and far right (x=+1)
    leftY = ww[0] + ww[1]*(-1.0)
    rightY = ww[0] + ww[1]*(1.0)

    wDict["x"].append(-1.0)
    wDict["yp"].append(leftY)
    wDict["x"].append(1.0)
    wDict["yp"].append(rightY)
    
    # Convert to a dataframe so it can be plotted like the other data
    resultW = pd.DataFrame(wDict)

    # Plot the corresponding classification separating line
    plt.plot(resultW.x, resultW.yp)


    plt.draw()
    plt.pause(delay)
    plt.clf()
    
def getEin(ww, data):
    Ein = 0
    # Sum the error for each datapoint to get the overall in sample error, Ein
    for index in range(0,len(data)):
        h = ww[0] + ww[1]*data["x"][index]
        Ein += (h - data["yp"][index])*(h - data["yp"][index])
    
    # Needs to be scaled by the number of datapoints
    Ein = Ein/len(data)

    return Ein
    
    
def regression(data):

    # Copy the initial weights so we can change them. Copy by value so they dont point to the same thing
    w = copy.deepcopy(initial_w)
    
    # How many times to we want to keep trying to update, configured by trial and error
    max_iterations = 300
    iteration = 0
    
    # How quickly should the weights update, configured by trial and error
    learning_rate = 0.5
    
    # Initial values so we can store the best fit
    bestW = copy.deepcopy(initial_w)
    minEin = getEin(initial_w, data)
    
    while iteration < max_iterations:
        # Choose a random datapoint to analyze
        index = numpy.random.randint(0, len(data))
        
        print("Iteration: {0}, index {1}, w {2}".format(iteration, index, w))
        
        # Plotting is the slowest part of each iteration. We want to see it updating, so lets 
        # just check after every 10 updates
        if iteration%10 == 0:
            plotCurrentW(w, data, 0.001)
        
        # Based on our weights, predict the y value
        h = w[0] + w[1]*data["x"][index]
        
        # Get the current in sample error for these weights
        Ein = getEin(w, data)
        
        if Ein < minEin:
            # Least error seen so far
            minEin = Ein
            bestW = copy.deepcopy(w)
           
        # Create a vector so that we can do multiplication with the weight vector more easily
        x = [1, data['x'][index]]
        
        # Derivative is 2x(h-y) for each datapoint, scaled by total number of points
        derivative = [2*elem*(h - data["yp"][index])/len(data) for elem in x] # Use a lambda function, could be done in a regular loop
        
        # Update the weights
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

