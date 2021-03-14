import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sn
import sklearn.preprocessing as sk


# takes in a datapoint and the array of centroids (2xnumC), returns index of winning node
# does i.w
def getDist(pt, centroids):
    dist = np.matmul(pt, np.transpose(centroids))
    return dist

# gets euclidean distance, not used
def getEuc(pt, centroids):
    num = len(centroids)
    dists = np.zeros(num)
    for i in range(num):
        dists[i] = np.linalg.norm(pt - centroids[i])
    return dists

# calculates squared error
def err(pt):
    return np.sum(np.square(pt))

# takes in data, number of centroids, and learning rate
# data is x, y coords (n x 2 matrix)
# applies clustering to find numC clusters
# returns cluster weights and classification of each datapoint
def KH1(data, numC, a, epochs=1000):
    eps = -1/numC
    features = len(data[0])
    num = len(data)
    maximums = np.max(data, 0)
    centroids = np.random.rand(numC, features)*maximums
    error = 100
    count = 0
    while error > 0.8 and count < epochs:
        count += 1
        # keep track of weight changes, if below a certain level, done
        # will also kick in if a causes error to get too small, since dependent -> num iterations
        error = 0
        wins = np.zeros((num, 1))
        # win is now nx1, with each element being the winning node
        # move the centroids that won in that direction
        for i in range(num):
            # get distance, max is closest (activation function? I guess?)
            dist = getDist(data[i], centroids)
            # use maxnet
            outputs = np.copy(dist)
            while np.sum(outputs) != np.max(outputs):
                for j in range(numC):
                    outputs[j] = np.max([0, outputs[j] + eps * (sum(outputs) - outputs[j])])
            winner = np.argmax(outputs) # get only non-zero element left

            wins[i] = winner
            # find datapoint - winnerCentroid
            temp = a*(data[i] - centroids[winner])
            centroids[winner] += temp # only increment winning centroid
            error += np.abs(err(temp)) # add to error tracker, when small enough, stop
        # reduce learning rate
        a = a*0.99

    print("Epochs and error for KH1:")
    print(count, error)
    return centroids, wins

# takes in data reduced by PCA to 3 features, num of centroids, learning rate
# applies clustering, returns centroid weights and classification of each datapoint
def KH2(data, numC, a, epochs=1000):
    eps = -1 / numC
    features = len(data[0])
    num = len(data)
    # randomly generate centroid weights as random points from the dataset
    # each row is a centroid, w 4 features
    maximums = np.max(data, 0)
    centroids = np.random.rand(numC, features)*maximums
    error = 100
    count = 0
    while error > 0.1 and count < epochs:
        count += 1
        error = 0

        wins = np.zeros((num, 1))
        # win is now nx1, with each element being the winning node
        # move the centroids that won in that direction
        for i in range(num):
            # applies i.w, max is closest
            dist = getDist(data[i], centroids)
            outputs = np.copy(dist)
            # max net
            while np.sum(outputs) != np.max(outputs):
                for j in range(numC):
                    outputs[j] = np.max([0, outputs[j] + eps * (sum(outputs) - outputs[j])])
            winner = np.argmax(outputs)
            wins[i] = winner
            # find datapoint - winnerCentroid
            temp = a * (data[i] - centroids[winner])
            centroids[winner] += temp
            error += np.abs(err(temp))

        a = a * 0.99
    print("Epochs and error for PCA-KH2")
    print(count, error)
    return centroids, wins


# load comma separated files as array
# returns array of data and answers
# each row corresponds to 1 datatype
# returns as x and y data separately - y is still in original format (strings)
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
# applies PCA, returns transformed data with reduced dimension
def pca(xtrain, num):
    ave = np.mean(xtrain, 0)
    xtrain = xtrain-ave
    # randomly initialize weights
    rows = np.size(xtrain,0)
    cols = np.size(xtrain,1)
    n = 0.1
    # will have 4 output nodes
    w = np.random.rand(num, cols)
    change = 100
    # decided based on tuning
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
    return w

# takes in actual answers and predicted answers
# assigns clusters to class labels
# returns clusters in order of classes and confusion matrix (out of order still)
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
    c[0] = np.argmax(temp[:,0])
    # set to -1 so it won't be chosen again
    temp[:, c[0]] = -1
    c[1] = np.argmax(temp[:,1])
    temp[:,c[1]] = -1
    c[2] = np.argmax(temp[:,2])
    return c, matrix # c now corresponds to 0, 1, and 2 from answers

# graphs points and centroids from KH1
# the 1st, 2nd, and 4th features are the coordinates and the 3rd feature is the colour
# centroids are red
def graph(xtrain, centroids):
    # plot for visualization
    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    ax.scatter(xtrain[:, 0], xtrain[:, 1], xtrain[:, 3], c=xtrain[:, 2])
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 3], c='r')
    plt.title("Train Data and Centroids")
    plt.show()

# graphs data and centroids from KH2-PCA
# takes in data, answers, and centroids
# uses answer as colour
def graph2(xtrain, y, centroids):
    # plot for visualization
    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    ax.scatter(xtrain[:, 0], xtrain[:, 1], xtrain[:, 2], c=y)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='r')
    plt.title("Train Data and Centroids")
    plt.show()

# takes in data and matrix made with pca
# applies pca to dataset
def applyPCA(xtrain, w):
    newX = np.matmul(xtrain, np.transpose(w))
    return newX

# finds how well the clusters classify test data
# takes in data, answers, and centroids
# returns confusion matrix
def test(x, y, c):
    num = len(x)
    predicted = np.zeros(num)
    for i in range(num):
        # find closest centroid
        dist = getEuc(x[i], c)
        predicted[i] = np.argmin(dist)

    # compare predicted and actual, make matrix
    matrix = np.zeros((3,3))
    for i in range(num):
        matrix[int(predicted[i]), y[i]] += 1

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    plt.figure(figsize=(10, 10))
    axis = plt.axes()
    sn.heatmap(matrix.round(1), annot=True, ax=axis, linewidths=0.5, xticklabels=classes, yticklabels=classes)

    plt.title("Confusion Matrix")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
    return matrix

# replaces y data with numbers
# just takes in ytrain
# NOTE, REQUIRES TXT FILE HAVE \n AT THE END OF THE FILE TOO
def convert(ytrain):
    classes = ['Iris-setosa\n', 'Iris-versicolor\n', 'Iris-virginica\n']
    num = len(ytrain)
    for i in range(num):
        for j in range(3):
            if ytrain[i] == classes[j]:
                ytrain[i] = j
                break
    ytrain = ytrain.astype(int).reshape(-1, 1)
    return ytrain


if __name__ == "__main__":
    file = "iris_train.txt"
    # get data
    xtrain, ytrain = loadFile(file)
    xtrain = sk.scale(xtrain)
    a = 0.8 # learning rate
    # find centroids
    # send in training data, number of centroids, learning rate (can also send in number of epochs)
    centroids, wins = KH1(xtrain, 3, a)

    # for each datapoint, find which cluster it belongs to and compare to which it was classified as
    num = len(wins) # this is number of datapoints

    # replace ytrain with ints (in order of classes)
    ytrain = convert(ytrain)

    c, matrix = assign(wins, ytrain)
    # reorder matrix and centroid weights to match classes
    matrix[[0,1,2]] = matrix[[c[0], c[1], c[2]]]
    centroids[[0,1,2]] = centroids[[c[0], c[1], c[2]]]
    print("Centroid weights in order of Iris-setosa, Iris-versicolor, and Iris-virginica (rows)")
    print(centroids)
    acc = np.trace(matrix)/np.sum(matrix)
    print("Train Accuracy:")
    print(acc)
    graph(xtrain, centroids)

    # given centroids, find out how test data is classified
    xtest, ytest = loadFile("iris_test.txt")
    xtest = sk.scale(xtest)
    ytest = convert(ytest)
    test(xtest, ytest, centroids)

    w = pca(xtrain, 3)
    print("PCA matrix:")
    print(w)
    # apply to both train and test data
    newX = applyPCA(xtrain, w)
    newX2 = applyPCA(xtest, w)

    # save to files
    np.savetxt("train1.txt", newX)
    np.savetxt("test1.txt", newX2)

    # send through other network
    centroids, wins = KH2(newX, 3, a)

    c, matrix = assign(wins, ytrain)
    # reorder
    matrix[[0,1,2]] = matrix[[c[0], c[1], c[2]]]
    centroids[[0,1,2]] = centroids[[c[0], c[1], c[2]]]
    print("Centroid weights in order of Iris-setosa, Iris-versicolor, and Iris-virginica (rows)")
    print(centroids)

    acc = np.trace(matrix) / np.sum(matrix)
    print("Train Accuracy:")
    print(acc)
    # uses ytrain as colour
    graph2(newX, ytrain, centroids)

    test(newX2, ytest, centroids)