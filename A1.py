# Jacqueline Heaton
# 16jh12
# CISC 874 Assignment 1
# contains code for both model 1 (from scratch) and model 2 (using keras)

from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import random
import math
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import plotly.graph_objects as go

# the sigmoid function was used as the activation function
# takes in a single number
# returns value of that number after it's passed through the sigmoid function
def sigmoid(x):
    return 1/(1 + np.e**-x)

# backpropagation function
# takes in weights from input to hidden layer, weights from hidden layer to output layer, the input data,
# the values of the hidden nodes currently, the predicted and actual outputs, the 2 biases, the learning rate,
# a, which is used for momentum, and the weights used for momentup

# Returns the 2 weight matrices, the 2 biases, and the 2 gradients from before the adjustment to be used next time
# for momentum
def backprop(inputWeights, hiddenWeights, initial, hiddenNodes, output, expected, b1, b2, rate, a, prevInput, prevHidden):
    # backpropagate from output to hidden layer
    # find difference between actual and expected output, since reused a lot
    diff = output - expected
    # 1st col is for hidden node x
    # this is the gradient and bias between hidden and output layers
    grad2 = (diff*output*(1-output)).reshape(-1,1)* hiddenNodes
    bias2 = np.sum(diff * output*(1-output))

    # sum of errors
    sumError = np.dot(diff*output*(1-output), hiddenWeights)
    # this is the gradient and bias between input and hidden layers
    grad1 = (sumError*hiddenNodes*(1-hiddenNodes)).reshape(-1,1)*initial
    bias1 = np.sum(sumError*hiddenNodes*(1-hiddenNodes))

    # save the weight change for next iteration for momentum
    newPrevInput = np.multiply(rate, grad1.T)
    newPrevHidden = np.multiply(rate, grad2)

    # update the weights with momentum (no momentum is just a=0)
    inputWeights = np.subtract(inputWeights, np.multiply(rate, grad1.T)) + a*prevInput
    hiddenWeights = np.subtract(hiddenWeights, np.multiply(rate, grad2)) + a*prevHidden
    b1 = b1 - rate * bias1
    b2 = b2 - rate * bias2

    # return all weights, biases, and the previous weight change
    return inputWeights, hiddenWeights, b1, b2, newPrevInput, newPrevHidden


# model made from scratch, prints results of test set
# takes in train and test data, number of nodes in hidden layer, number of epochs and a for momentum
def model1(xTrain, yTrain, xTest, yTest, hidden, maxIter, a):
    # total number of images
    numImages = len(xTrain)

    # images are input, has 784 pixels
    x = len(xTrain[0])
    # number of output nodes
    out = 10

    # initialize weights randomly
    # randomly gerenated weights from -1 to 1
    # from input to hidden layer
    inputWeights = np.random.uniform(-1, 1, size=(x,hidden))
    # from hidden to output
    hiddenWeights = np.random.uniform(-1, 1, size=(out, hidden))

    # initialize momentum arrays
    # just want the size, start with 0
    prevInput = np.zeros((x, hidden))
    prevHidden = np.zeros((out,hidden))

    # initialize biases, learning rate, and momentum fraction
    b1 = 1
    b2 = 1
    rate = 0.1

    # max number of iterations/epochs and counter
    iter = 0
    # take in as input instead
    # maxIter = 30

    # store max accuracy value so that you can save best weights
    maxAccuracy = 0
    # iterate until loss is down or maxIter hit
    aveError = 1 # start with error of 100%, will be overwritten soon
    # once error hits less than 0.02, tends to not improve much more on test dataset
    while aveError > 0.02 and iter < maxIter:
        # index tracker for what image we're on
        count = 0
        aveError = 0
        # store whether image was classified correctly or not as binary 0/1 (1 is correct, 0 is not)
        accuracy = [0 for _ in range(numImages)]
        for i in xTrain:
            # forward propagation
            # use normalized data, since get much better accuracies
            hiddenNodes = np.array(list(map(sigmoid, np.dot(preprocessing.normalize([i]).reshape(784,), inputWeights) + b1)))
            outputNodes = np.array(list(map(sigmoid, np.dot(hiddenNodes, hiddenWeights.T) + b2)))

            # use back prop to adjust inputWeights and hiddenWeights
            # desired output

            # get desired output in one-hot-encoding form
            expected = [0 for i in range(out)]
            d = int(yTrain[count])
            expected[d] = 1

            # calculate error
            aveError += np.sum(0.5 * (outputNodes - expected) ** 2)

            # do backprop
            inputWeights, hiddenWeights, b1, b2, prevInput, prevHidden = backprop(
                inputWeights, hiddenWeights, i, hiddenNodes, outputNodes, expected, b1, b2, rate, a, prevInput, prevHidden)

            # find whether classified correctly or not, print accuracy for each for loop
            # returns array for some reason
            temp = np.where(outputNodes == np.amax(outputNodes))[0]
            if temp == d:
                accuracy[count] = 1
            count += 1
        # calculate the average error for this epoch
        aveError = aveError/numImages
        # increase epoch tracker counter
        iter += 1
        print("Accuracy:", np.average(accuracy))
        # save weights at end of epoch if they get the best accuracy we've seen
        if np.average(accuracy) > maxAccuracy:
            bestInput = inputWeights
            bestHidden = hiddenWeights
            maxAccuracy = np.average(accuracy)
        print("Error:", aveError)
        print()

    # apply weights to test data, use best weights
    numImages = len(xTest)
    # for getting stats on training data instead, uncomment the next 3 lines:
    # xTest = xTrain
    # yTest = yTrain
    # numImages = len(xTrain)

    count = 0
    num = 0
    print("Epochs:", iter)
    output = [0 for _ in range(numImages)]

    for i,d in zip(xTest, yTest):
        hiddenNodes = np.array(list(map(sigmoid, np.dot(preprocessing.normalize([i]).reshape(784,), bestInput) + b1)))
        outputNodes = np.array(list(map(sigmoid, np.dot(hiddenNodes, bestHidden.T) + b2)))
        ans = np.where(outputNodes == np.amax(outputNodes))[0]
        output[num] = ans
        num += 1
        # calculate how many were right
        if ans == int(d):
            count += 1

    print("Test Accuracy:", count/numImages)

    # for confusion matrix
    data = {'y_Actual': list(map(int, yTest)),
            'y_Predicted': list(map(int,output))}

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'],
                                   margins=True)
    plt.figure(figsize=(10,10))
    axis = plt.axes()
    table = tabulate(confusion_matrix, headers=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Total'])
    print(table)

    # calculate precision and recall
    precision = [[0] for _ in range(10)]
    recall = [[0] for _ in range(10)]
    tp = [[0] for _ in range(10)]
    fp = [[0] for _ in range(10)]
    fn = [[0] for _ in range(10)]
    for i in range(10):
        tp[i][0] = confusion_matrix[i][i]
        fp[i][0] = confusion_matrix['All'][i] - tp[i][0]
        fn[i][0] = confusion_matrix[i]['All'] - tp[i][0]
        precision[i][0] = tp[i][0]/(tp[i][0]+fp[i][0])
        recall[i][0] = tp[i][0] / (tp[i][0] + fn[i][0])

    stat = go.Figure(data=go.Table(header=dict(values=['Numbers','True Positives', 'False Positives','False Negatives', 'Precision','Recall']),
                                cells=dict(values=[[[i] for i in range(10)], tp, fp, fn, precision, recall])))
    # this will open up a internet browser
    # stat.show()

    # makes a heatmap of the confusion matrix
    sn.heatmap(confusion_matrix.round(1), annot=True, ax=axis, linewidths=0.5)
    plt.title("Confusion Matrix for " + str(hidden) + " Hidden Nodes")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    # I'm assuming you don't want this printed
    # plt.show()


# Neural net using keras libraries
# takes in test and train data, number of nodes in hidden layer, and number of epochs
def model2(xTrain, yTrain, xTest, yTest, hidden, maxIter):
    # build model, describe input, use number of hidden nodes specified
    # output is 10 nodes
    # both use sigmoid activation function
    model = Sequential([
        Dense(hidden, activation='sigmoid', input_shape=(784,)),
        Dense(10, activation='sigmoid',)
    ])

    # compile model, use adam optimizer, categorical crossentropy, and assess using accuracy
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # fit/train using the train data
    # set epochs to 30 because we used that for model1
    # set verbose to 0 because it prints too much stuff
    model.fit(
        xTrain,
        to_categorical(yTrain),
        epochs=maxIter,
        verbose=0,
    )

    # this is the equivalent of accuracy, which I didn't wanna calculate. That's all this is used for.
    score = model.evaluate(xTest, to_categorical(yTest), verbose=0)
    print(score)

    # get actual predictions so I can get the confusion matrix and precision and recall
    # for training data stats, uncomment the next 2 lines:
    # xTest = xTrain
    # yTest = yTrain

    output = model.predict(
        xTest
    )
    predicted = np.argmax(output,1)

    data = {'y_Actual': list(map(int, yTest)),
            'y_Predicted': predicted}

    # make comfusion matrix
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'],
                                   margins=True)
    plt.figure(figsize=(10, 10))
    axis = plt.axes()
    table = tabulate(confusion_matrix, headers=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Total'])
    print(table)

    # calculate precision and recall
    precision = [[0] for _ in range(10)]
    recall = [[0] for _ in range(10)]
    tp = [[0] for _ in range(10)]
    fp = [[0] for _ in range(10)]
    fn = [[0] for _ in range(10)]
    for i in range(10):
        tp[i][0] = confusion_matrix[i][i]
        fp[i][0] = confusion_matrix['All'][i] - tp[i][0]
        fn[i][0] = confusion_matrix[i]['All'] - tp[i][0]
        precision[i][0] = tp[i][0] / (tp[i][0] + fp[i][0])
        recall[i][0] = tp[i][0] / (tp[i][0] + fn[i][0])

    stat = go.Figure(data=go.Table(
        header=dict(values=['Numbers', 'True Positives', 'False Positives', 'False Negatives', 'Precision', 'Recall']),
        cells=dict(values=[[[i] for i in range(10)], tp, fp, fn, precision, recall])))

    # stat.show()

    sn.heatmap(confusion_matrix.round(1), annot=True, ax=axis, linewidths=0.5)
    plt.title("Confusion Matrix for " + str(hidden) + " Hidden Nodes")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    #plt.show()


if __name__ == '__main__':
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=60000, test_size=10000, shuffle=False)

    # 20, 30, 40, 50, 60, 70, 80, 90, 100
    # you can fill this array with whatever you want
    for hidden in [70]:
        print("HIDDEN:", hidden)
        # this is the momentum multiplier, set to 0 for no momentum
        a = 0.5
        epoch = 30
        model1(xTrain, yTrain, xTest, yTest, hidden, epoch, a)
        # have to preprocess data here for model 2
        model2(list(map(preprocessing.normalize, [xTrain]))[0], yTrain, list(map(preprocessing.normalize, [xTest]))[0], yTest, hidden, epoch)



