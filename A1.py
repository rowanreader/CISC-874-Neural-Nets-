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

# Got math from:
# https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/?fbclid=IwAR1lRXZCdZIxOSyQal0vtIwaSvCTzVwKfSiJ2UDMFmsn2Oocux0yjAX3shw

def sigmoid(x):
    return 1/(1 + np.e**-x)

def dSigmoid(x):
    return np.e**-x/(np.e**-x + 1)**2

def backprop(inputWeights, hiddenWeights, initial, hiddenNodes, output, expected, b1, b2, rate):
    # backpropagate from output to hidden layer
    # find difference between actual and expected output, since reused a lot
    diff = output - expected
    # 1st col is for hidden node x
    grad2 = (diff*output*(1-output)).reshape(-1,1)* hiddenNodes
    bias2 = np.sum(diff * output*(1-output))

    # sum of errors
    sumError = np.dot(diff*output*(1-output), hiddenWeights)
    grad1 = (sumError*hiddenNodes*(1-hiddenNodes)).reshape(-1,1)*initial
    bias1 = np.sum(sumError*hiddenNodes*(1-hiddenNodes))


    inputWeights = np.subtract(inputWeights, np.multiply(rate, grad1.T))
    hiddenWeights = np.subtract(hiddenWeights, np.multiply(rate, grad2))
    b1 = b1 - rate * bias1
    b2 = b2 - rate * bias2
    return inputWeights, hiddenWeights, b1, b2


# model made from scratch, prints results of test set
# takes in train and test data, and number of nodes in hidden layer
def model1(xTrain, yTrain, xTest, yTest, hidden):
    # total number of images
    numImages = len(xTrain)

    # images are input, has 784 pixels
    x = len(xTrain[0])
    # number of output nodes
    out = 10

    # randomly gerenated weights from -1 to 1
    # from input to hidden layer
    inputWeights = np.random.uniform(-1, 1, size=(x,hidden))
    # from hidden to output
    hiddenWeights = np.random.uniform(-1, 1, size=(out, hidden))
    b1 = 1
    b2 = 1
    rate = 0.01

    # multiply image through then apply back prop
    iter = 0
    maxIter = 30
    maxAccuracy = 0
    # iterate until loss is down or maxIter hit
    aveError = 1 # start with error of 100%
    # once error hits less than 0.03, tends to not improve much more on test dataset
    while aveError > 0.03 and iter < maxIter:
        count = 0
        aveError = 0
        accuracy = [0 for _ in range(numImages)]
        for i in xTrain:
            hiddenNodes = np.array(list(map(sigmoid, np.dot(preprocessing.normalize([i]).reshape(784,), inputWeights) + b1)))
            outputNodes = np.array(list(map(sigmoid, np.dot(hiddenNodes, hiddenWeights.T) + b2)))

            # use back prop to adjust inputWeights and hiddenWeights
            # desired output

            expected = [0 for i in range(out)]
            d = int(yTrain[count])
            expected[d] = 1

            aveError += np.sum(0.5 * (outputNodes - expected) ** 2)

            inputWeights, hiddenWeights, b1, b2 = backprop(inputWeights, hiddenWeights, i, hiddenNodes, outputNodes, expected, b1, b2, rate)

            # find whether classified correctly or not, print accuracy for each for loop
            # returns array for some reason
            if np.where(outputNodes == np.amax(outputNodes))[0] == d:
                accuracy[count] = 1
            count += 1
        aveError = aveError/numImages
        iter += 1
        print("Accuracy:", np.average(accuracy))
        if np.average(accuracy) > maxAccuracy:
            bestInput = inputWeights
            bestHidden = hiddenWeights
            maxAccuracy = np.average(accuracy)
        print("Error:", aveError)
        print()

    # here, go to test, use best weights
    numImages = len(xTest)
    count = 0
    num = 0
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
    # data = {'y_Actual': list(np.ones(numImages)),
    #         'y_Predicted': output}

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
    # stats = tabulate([[[i] for i in range(10)], tp, fp, fn, precision, recall],
    #                  headers=['Numbers','True Positives', 'False Positives','False Negatives', 'Precision','Recall'])
    # print(stats)
    stat.show()

    sn.heatmap(confusion_matrix.round(1), annot=True, ax=axis, linewidths=0.5)
    plt.title("Confusion Matrix for " + str(hidden) + " Hidden Nodes")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def model2(xTrain, yTrain, xTest, yTest, hidden):
    model = Sequential([
        Dense(hidden, activation='sigmoid', input_shape=(784,)),
        Dense(10, activation='sigmoid',)
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        xTrain,
        to_categorical(yTrain),
        epochs=30,
        verbose=0,
    )

    score = model.evaluate(xTest, to_categorical(yTest), verbose=0)
    print(score)

    output = model.predict(
        xTest
    )
    predicted = np.argmax(output,1)

    data = {'y_Actual': list(map(int, yTest)),
            'y_Predicted': predicted}


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

    #stat.show()

    sn.heatmap(confusion_matrix.round(1), annot=True, ax=axis, linewidths=0.5)
    plt.title("Confusion Matrix for " + str(hidden) + " Hidden Nodes")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    #plt.show()


if __name__ == '__main__':
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=60000, test_size=10000, shuffle=False)
    # xTrain = 0
    # yTrain = 0
    # xTest = 0
    # yTest = 0
    # 20, 30, 40, 50, 60, 70, 80, 90, 100
    for hidden in [20, 30, 40, 50, 60, 70, 80, 90, 100,110,120]:
        print("HIDDEN:", hidden)
        model1(xTrain, yTrain, xTest, yTest, hidden)
        # model2(list(map(preprocessing.normalize, [xTrain]))[0], yTrain, list(map(preprocessing.normalize, [xTest]))[0], yTest, hidden)
        # model2(xTrain, yTrain, xTest,
        #        yTest, hidden)



