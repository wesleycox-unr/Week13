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

# Number of data points
N = 200


def getRand():
    return numpy.random.uniform(-1,1)

def getY(xVal,m,b):

    lineY = m*xVal + b
    
    std_dev = 0.2
    
    y = numpy.random.normal(lineY, std_dev)

    return y

def plotData(currentData, delay):
    # Plot the randomly generated data
    plt.scatter(currentData["x"],currentData["yp"], marker="o")
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.draw()
    plt.pause(delay)
    plt.clf()
    
    
def createData(m, b):

    dataDict = {"x": [], "yp":[]}

    # Create N datapoints
    for i in range(N):
        x = getRand()
        y = getY(x, m, b)

        dataDict['x'].append(x)
        dataDict['yp'].append(y)

    data = pd.DataFrame(dataDict)
    return data

line_gradient = 1 # 45 degree line
intercept = 0 # passing through the origin
input_data = createData(line_gradient,intercept)

print(input_data)

input_data.to_json("linear_data_1D.txt")

plotData(input_data, 1)