import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import mode
import random


# takes in a datapoint and the array of centroids (2xnumC), returns index of winning node
def getDist(pt, centroids):
    dist = np.matmul(pt, np.transpose(centroids))
    # result = np.where(dist == np.amax(dist))
    return dist

def getEuc(pt, centroids):
    dists = np.zeros(len(centroids))
    num = len(centroids)
    for i in range(num):
        dists[i] = np.linalg.norm(pt - centroids[i])
    return dists

def err(pt):
    return np.sum(np.square(pt))

# takes in data, number of centroids, and learning rate
# data is x, y coords (n x 2 matrix)
def KH1(data, numC, a, epochs=1000):
    eps = -1/numC
    features = len(data[0])
    num = len(data)

    # normalize data across rows
    sums = data.sum(axis=1) # sum across colums
    normData = data/sums[:, np.newaxis]

    # randomly generate centroid weights as random points from the dataset
    # each column is a centroid, w 4 features
    c = random.sample(range(1,num), 3)
    # centroids = np.array([data[i] for i in c])
    maximums = np.max(data, 0)
    minimums = np.min(data, 0)
    # diff = maximums - minimums
    centroids = np.random.uniform(low=minimums, high=maximums, size=(numC, features))
    error = 100
    count = 0
    while error > 1 and count < epochs:
        count += 1
        # to find closest centroid to data, find max i*w
        # dist = np.matmul(data, centroids)
        # dist should be n x 3, each row is a point with the distances to the 3 centroids
        # get max index for each row
        # win = np.argmax(dist, 1)
        # keep track of weight changes, if below a certain level, done
        # will also kick in if a gets too small, since dependent -> num iterations
        error = 0

        wins = np.zeros((num, 1))
        # win is now nx1, with each element being the winning node
        # move the centroids that won in that direction
        for i in range(num):
            # get normalized centroids
            # sum across cols
            sums = centroids.sum(axis=0)
            normCentroids = centroids / sums[np.newaxis, :]

            # get distance, max is closest (activation function? I guess?)
            # dist = getDist(normData[i], normCentroids)
            dist = getEuc(data[i], centroids)

            winner = np.argmin(dist)
            # winner = np.argmax(dist)
            wins[i] = winner
            # find datapoint - winnerCentroid
            temp = a*(data[i] - centroids[winner])
            centroids[winner] += temp
            error += np.abs(err(temp))

        a = a*0.99
    print(count, error)
    return centroids, wins

def KH2(data, numC, a, epochs=1000):
    eps = -1 / numC
    features = len(data[0])
    num = len(data)

    # normalize data across rows
    sums = data.sum(axis=1)  # sum across colums
    normData = data / sums[:, np.newaxis]

    # randomly generate centroid weights as random points from the dataset
    # each column is a centroid, w 4 features
    c = random.sample(range(1, num), 3)
    # centroids = np.array([data[i] for i in c])
    maximums = np.max(data, 0)
    minimums = np.min(data, 0)
    # diff = maximums - minimums
    centroids = np.random.uniform(low=minimums, high=maximums, size=(numC, features))
    error = 100
    count = 0
    while error > 0.5 and count < epochs:
        count += 1
        # to find closest centroid to data, find max i*w
        # dist = np.matmul(data, centroids)
        # dist should be n x 3, each row is a point with the distances to the 3 centroids
        # get max index for each row
        # win = np.argmax(dist, 1)
        # keep track of weight changes, if below a certain level, done
        # will also kick in if a gets too small, since dependent -> num iterations
        error = 0

        wins = np.zeros((num, 1))
        # win is now nx1, with each element being the winning node
        # move the centroids that won in that direction
        for i in range(num):
            # get normalized centroids
            # sum across cols
            sums = centroids.sum(axis=0)
            normCentroids = centroids / sums[np.newaxis, :]

            # get distance, max is closest (activation function? I guess?)
            # dist = getDist(normData[i], normCentroids)
            dist = getEuc(data[i], centroids)

            winner = np.argmin(dist)
            # winner = np.argmax(dist)
            wins[i] = winner
            # find datapoint - winnerCentroid
            temp = a * (data[i] - centroids[winner])
            centroids[winner] += temp
            error += np.abs(err(temp))

        a = a * 0.99
    print(count, error)
    return centroids, wins


# load comma separated files as array
# returns array of data and answers
# each row corresponds to 1 datatype
def loadFile(file):
    data = open(file, 'r')
    data = data.readlines()
    num = len(data) # how many points are in each file
    for i in range(num):
        data[i] = data[i].split(",")
    data = np.array(data)

    features = len(data[0]) # there are n-1 features, the last one is the answer
    x = data[:, 0:features-1]
    x = x.astype(np.float)
    y = data[:, features-1]
    return x, y

# takes in training data and how many features to reduce to
def pca(xtrain, num):
    # num = 1
    # xtrain = np.array([[1.3, 3.2, 3.7], [1.4, 2.8, 4.1], [1.5, 3.1, 4.6], [1.2, 2.9, 4.8], [1.1, 3, 4.8]])
    ave = np.mean(xtrain, 0)
    xtrain = xtrain-ave
    eps = 0.1 # set epsilon for threshold
    # randomly initialize weights
    rows = np.size(xtrain,0)
    cols = np.size(xtrain,1)
    n = 0.15
    # will have 4 output nodes
    w = np.random.rand(num, cols)
    # w = np.array([0.3, 0.4, 0.5])
    norm = np.linalg.norm(w)
    change = 100
    while change > 7:
        change = 0
        # for each datapoint
        for i in range(rows):
            y = np.dot(w, xtrain[i])
            # for each feature in the weight vector
            for j in range(num):
                temp = n*y[j]*(xtrain[i] - np.dot(y,w))
                w[j, :] += temp
                change += np.linalg.norm(temp)
                # terminate when |w| reaches close to 1
        norm = np.linalg.norm(w)
    print(norm)
    return w


# assigns clusters to class labels
# returns clusters in order of classes
def assign(predicted, answers):
    # make 3x3 matrix to store accuracies
    # row is predicted, col is actual
    matrix = np.zeros((3,3))
    for i in range(3):
        temp = np.where(predicted == i)[0]
        for j in temp: # essentially counts how many of each of the actual classes were in the predicted class
            matrix[i][answers[j][0]] += 1
    c = np.array([0,0,0])
    # get max for 1st predicted class (row 1)
    temp = np.copy(matrix)
    c[0] = np.argmax(temp[0])
    # set to -1 so it won't be chosen again
    temp[:, c[0]] = -1
    c[1] = np.argmax(temp[1])
    temp[:,c[1]] = -1
    c[2] = np.argmax(temp[2])
    return c, matrix # c now corresponds to 0, 1, and 2 from answers

def graph(xtrain, centroids):
    # plot for visualization
    fig = plt.figure()
    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    ax.scatter(xtrain[:, 0], xtrain[:, 1], xtrain[:, 3], c=xtrain[:, 2])

    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 3], c='r')
    # plt.plot(xtrain[:, 0], xtrain[:, 1], '.')
    # plt.plot(centroids[0,:], centroids[1,:], 'o')
    plt.show()

def graph2(xtrain, y, centroids):
    # plot for visualization
    fig = plt.figure()
    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    ax.scatter(xtrain[:, 0], xtrain[:, 1], xtrain[:, 2], c=y)

    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='r')
    # plt.plot(xtrain[:, 0], xtrain[:, 1], '.')
    # plt.plot(centroids[0,:], centroids[1,:], 'o')
    plt.show()

def applyPCA(xtrain, w):
    newX = np.matmul(xtrain, np.transpose(w))
    return newX

if __name__ == "__main__":
    file = "iris_train.txt"
    # file = "iris_test.txt"
    # get data
    xtrain, ytrain = loadFile(file)

    # find centroids
    # send in training data, number of centroids, learning rate (can also send in number of epochs)
    centroids, wins = KH1(xtrain, 3, 0.5)

    # for each datapoint, find which cluster it belongs to and compare to which it was classified as
    num = len(wins) # this is number of datapoints
    classes = ['Iris-setosa\n','Iris-versicolor\n','Iris-virginica\n']
    # replace ytrain with ints (in order of classes)
    for i in range(num):
        for j in range(3):
            if ytrain[i] == classes[j]:
                ytrain[i] = j
                break
    ytrain = ytrain.astype(int).reshape(-1, 1)
    c, matrix = assign(wins, ytrain)

    print(centroids)
    print(c)
    print(matrix)

    graph(xtrain, centroids)

    w = pca(xtrain, 3)
    print(w)

    newX = applyPCA(xtrain, w)

    centroids, wins = KH2(newX, 3, 0.5)
    print(centroids)
    graph2(newX, ytrain, centroids)