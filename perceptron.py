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
N = 20

#    w0    w1  w2
initial_w = [0.1, 0.1, 0.1]

def getRand():
    return numpy.random.uniform(-1,1)

def sign(x1,x2,m,b):
    # If the x1 and x2 position lies above the line corresponding to m and b, then it is positive, otherwise negative
    liney = m*x1 + b

    if x2 > liney:
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

def perceptron(data):

    # Implement the perceptron learning algorithm to determine the weights w that will fit the data
    w = initial_w
    
    #print(data)
    
    #for row in data.iterrows():
    index = 0
    while index < len(data):
        print("Index {0} of {1}".format(index, len(data)))
        
        row = data.iloc[index]
        
        #print(row)
        #sys.exit()
        #print(row[1]["x1"])
        h = w[0]*1 + w[1]*row["x1"] + w[2]*row["x2"]
        
        if numpy.sign(h) != numpy.sign(row["yp"]):
            # Mismatch
            # Update w values
            
            #w[w0 w1 w2] = w + yp * data
            w[0] = w[0] + row["yp"]*1
            w[1] = w[1] + row["yp"]*row["x1"]
            w[2] = w[2] + row["yp"]*row["x2"]
            index = 0
        else:
            index += 1
    # Steps:
    # - See if the current weights will correctly predict the yp values in the DataFrame for all rows
    # - If so, done.
    # - If not, choose a row that isnt predicted correctly, and update the weights by the scalar product of yp with [1, x1, x2]
    # - Repeat until all rows are correctly predicted


    return w # Return the predicting weights

line_gradient = 1 # 45 degree line
intercept = 0 # passing through the origin
input_data = createData(line_gradient,intercept)

# Perform the learning
w = perceptron(input_data)

print(w)

# Plot the randomly generated data
plt.scatter(input_data[input_data["yp"] == 1]["x1"],input_data[input_data["yp"] == 1]["x2"], marker="o")
plt.scatter(input_data[input_data["yp"] == -1]["x1"],input_data[input_data["yp"] == -1]["x2"], marker="x")
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("x1")
plt.ylabel("x2")

# Create data points corresponding the the weights so that they can be plotted on the graph
wDict = {'x1':[], 'x2':[]}
leftx2 = (w[1] - w[0])/w[2]
rightx2 = (-w[1] - w[0])/w[2]

wDict["x1"].append(-1.0)
wDict["x2"].append(leftx2)
wDict["x1"].append(1.0)
wDict["x2"].append(rightx2)

resultW = pd.DataFrame(wDict)

# Plot the corresponding classification separating line
plt.plot(resultW.x1, resultW.x2)



plt.show()