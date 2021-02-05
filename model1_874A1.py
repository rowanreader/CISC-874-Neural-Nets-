from tensorflow.keras.datasets import mnist
import math
import numpy as np

def init_weights(layer_dims):
    """
    Randomly initialize the weight values for a neural network

    Arguments:
        layer_dims -- list containing the dimensions (num nodes) in each layer of the network

    Returns:
        parameters --  dictionary containing weight and bias values for each layer:
    """
    # store weights in parameters dictionary
    # includes bias weight for input and hidden layer
    parameters = {'w_10': np.random.uniform(-1, 1, size=(layers_dims[1],layers_dims[0])), #np.random.rand(layers_dims[1], layers_dims[0]),
                  'w_21': np.random.uniform(-1,1, size=(layers_dims[2], layers_dims[1])),
                  # 'b_10': np.random.rand(1) * 0.01,
                  # 'b_21': np.random.rand(1) * 0.01
                  'b_10': 1,
                  'b_21': 1
                  }


    # get number of layers to loop over
    # num_layers = len(layer_dims)
    # for l in range(1, num_layers):
    #     parameters['w_' + str(l) + str(l-1)] = np.random.randn(layer_dims[l], layer_dims[l-1]+1) * 0.01
    #     # parameters['b_' + str(l) + str(l-1)] = np.zeros((layer_dims[l], 1))

    return parameters

def weighted_sum(x, w, b):
    """
    Calculate the sum of x, weighted by w, and bias b

    Arguments:
        x -- input values
        w -- weights to apply
        b -- bias term

    Returns:
        a --  weighted sum of x and w plus bias
    """
    a = np.dot(w,x) + b
    return a

def sigmoid_activation(a):
    """
    Applying sigmoid function to activation value

    Arguments:
        a -- activation value(s) to input to sigmoid function

    Returns:
        y -- output of sigmoid function, numpy array
    """
    y = np.zeros(a.shape)
    # Two versions implemented for stability (preventing positive and negative overflow)
    for i, val in enumerate(a):
        y[i] = 1/(1 + np.e**(-val))
        # if val >= 0:
        #     y[i] = 1 / (1 + math.exp(-val))
        # else:
        #     y[i] = math.exp(val) / (1 + math.exp(val))

    return y


def train(X, d, layers_dims, epochs = 100, c = 0.001, verbose = 0):
    """
    Function for training a neural network with structure input, 1 hidden layer, 1 output

    Arguments:
        X -- training dataset: [# samples, dimension of input, dimension of input]
        d -- desired output values, corresponding to X: [# samples, ]
        epochs -- number of times to go through entire X set: float value, default 100
        c -- small constant learning rate: float value, default 0.001

    """

    # constants for while loop - could be inputs
    E_THRESHOLD = 0.001

    # Value for for loop range
    num_samples = X.shape[0]

    # initialize weights for network
    parameters = init_weights(layers_dims)

    # parameters = {'w_10': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
    #               'w_21': np.array([[0.7, 0.9], [0.8, 0.1]]),
    #               'b_10': np.array([0.5]),
    #               'b_21': np.array([0.5])
    # }


    # Convert d values to one hot encoding
    d_encode = np.zeros((d.shape[0],10))
    for i, val in enumerate(d):
        d_encode[i][val] = 1

    # Configure shape of d_encode to avoid empty shape value
    #d_encode = d_encode.reshape(d_encode.shape[0], d_encode.shape[1], 1)

    # epoch counter
    i = 0
    # initialize error value to enter while loop
    sum_sq_error = 10**8

    # while SSE is unsatisfactory and computational bounds are not exceeded
    while sum_sq_error > E_THRESHOLD and i < epochs:
        # reset error for each epoch
        sum_sq_error = 0
        # initialize array to store predicted values in
        y_pred = np.zeros(num_samples)

        # print("w_21 before: ", parameters['w_21'][0][0])
        # print("w_10 before: ", parameters['w_10'][0][0])

        # for each input pattern
        for p in range(0, num_samples):                   # Size of each step:
            # Removing None part of shape
            x_i = X[p]

            # compute activations at hidden nodes
            a_h = weighted_sum(x_i, parameters['w_10'], parameters['b_10'])  # (num hidden nodes, 1)

            # compute outputs from hidden nodes
            y_h = sigmoid_activation(a_h)                # (num hidden nodes, 1)

            # compute activations at output nodes
            a_j = weighted_sum(y_h, parameters['w_21'], parameters['b_21'])  # (num output nodes, 1)
            # compute network outputs
            y_j = sigmoid_activation(a_j)                # (num output nodes, 1)

            # calculate squared difference for error
            p_diff = (y_j - d_encode[p])
            sum_sq_error = sum_sq_error + sum(p_diff ** 2)

            # store prediction based on largest value in y_j
            y_pred[p] = np.argmax(y_j)

            # Back prop
            # Calculate update for weights between hidden & output nodes
            delta_w_jh = (p_diff * y_j * (1 - y_j)).reshape(-1,1)*np.transpose(y_h)                # [num output nodes, num hidden nodes + 1]
            # Calculate update for bias between hidden & output nodes
            delta_b_jh = sum(p_diff * y_j * (1 - y_j))

            # Calculate update for weights between input & hidden nodes
            delta = np.dot(p_diff * y_j * (1 - y_j), parameters['w_21'])
            # delta = delta.reshape(delta.shape[0],1)
            delta_w_hi = (delta * (y_h * (1 - y_h))).reshape(-1,1) * x_i

            # Calculate update for bias between input & hidden nodes
            # Get diagonal weight value going to the node in the same row
            delta_a = parameters['w_21']
            # delta_a = delta_a.reshape(delta_a.shape[0],1)
            delta_Ee = p_diff * (y_j * (1 - y_j))
            delta_Ee_a = np.dot(np.transpose(delta_Ee), delta_a)
            delta_b_hi = np.sum(delta_Ee_a * (y_h * (1 - y_h)))

            # print("delta w_jh: ", delta_w_jh)
            # print("delta b_jh: ", delta_b_jh)
            # print("w_21 before: ", parameters['w_21'][0][0])
            # print("w_10 before: ", parameters['w_10'][0][0])

            # Update weights between hidden & output nodes
            parameters['w_21'] = parameters['w_21'] - c * delta_w_jh          # [num output nodes, num hidden nodes + 1]
            # Update weights between input and & hidden nodes
            parameters['w_10'] = parameters['w_10'] - c * delta_w_hi          # [num hidden nodes, num input nodes + 1]

            # print("w_21 after: ", parameters['w_21'][0][0])
            # print("w_10 after: ", parameters['w_10'][0][0])

            # Update bias for hidden to output layer
            parameters['b_21'] = parameters['b_21'] - c * delta_b_jh
            # Update bias for input to hidden layer
            parameters['b_10'] = parameters['b_10'] - c * delta_b_hi

            # end for

        # print("w_21 after: ", parameters['w_21'][0][0])
        # print("w_10 after: ", parameters['w_10'][0][0])

        # Calculate accuracy for this epoch
        accuracy = round(len(np.where(y_pred == d)[0]) / num_samples * 100, 4)

        if verbose:
            print("Epoch ", i+1)
            print("Loss: ", np.round(sum_sq_error, 4))
            print("Accuracy: ", accuracy)

        # increment epoch counter
        i += 1
    # end while

    return parameters, sum_sq_error, accuracy

if __name__ == "__main__":
    # Load in data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_X = X_train.reshape(X_train.shape[0], -1)
    test_X = X_test.reshape(X_test.shape[0], -1)

    train_y = y_train.reshape(y_train.shape[0], 1)
    test_y = y_test.reshape(y_test.shape[0], 1)


    hidden_width = 32
    epochs = 10
    learning_rate = 0.01
    verbose = 1
    layers_dims = (train_X.shape[1], hidden_width, 10)
    model, error, accuracy = train(train_X, y_train, layers_dims, epochs, learning_rate, verbose)

    # inputs = np.array([[1, 4, 5]])
    #
    # targets = np.array([[0.1], [0.05]])
    # hidden_width = 2
    # epochs = 2
    # learning_rate = 0.01
    # verbose = 1
    # layers_dims = (inputs.shape[0], hidden_width, 2)
    # model, error, accuracy = train(inputs, targets, layers_dims, epochs, learning_rate, verbose)








# Old code
# delta = np.sum(np.matmul(np.transpose(p_diff * y_j * (1 - y_j)), parameters['w_21']))
#             print("Delta: ", delta)
#             delta_w_hi = c * delta * y_h * (1 - y_h) * np.transpose(x_i)                                # [num hidden nodes, num input nodes + 1]
#             print("Delta_w1: ", delta_w_hi[0])
