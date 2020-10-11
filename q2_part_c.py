import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def load_data(x_file,y_file):
    x = np.loadtxt(x_file)
    Y = np.loadtxt(y_file)
    #ensure compatible dimensionality
    X = x.reshape((200, 1))
    y = Y.reshape((200, 1))
    x = x.reshape((200, 1))

    # create vector of ones...
    int = np.ones(shape=Y.shape)[..., None]

    # ...and add to feature matrix
    X = np.concatenate((int, X), 1)
    return X,x,y

def empirical_risk(y,y_pred):
    #calculate empirical risk
    diff = y-y_pred
    square_loss = (diff**2) / 2
    empirical_risk = sum(square_loss)/len(y)
    return empirical_risk

def CFLR(X,y):
    # calculate coefficients using closed-form solution
    coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return coeffs

def closed_form_regress(X_bias, y):
    '''
    X_bias: features with bias, array
    y: labels, vector
    '''
    theta_best = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)
    return theta_best

def cal_cost(theta, X, y):
    '''
    theta: weights, vector
    X: feautes, array
    y: labels, vector
    '''
    m = len(y)
    prediction = X.dot(theta)
    cost = (1/(2.0*m))*np.sum(np.square(prediction-y))
    return cost

def polyRegress(x,y,d):
    # x is the raw data values
    x_matrix = x**0
    for power in range(1,d+1):
        x_matrix = np.append(x_matrix,x**power,axis=1)
    weights = CFLR(x_matrix,y)
    y_pred = x_matrix.dot(weights)
    em_loss = empirical_risk(y,y_pred)
    return em_loss[0],weights,y_pred


X,x,y = load_data("hw1x.dat","hw1y.dat")

for i in range(3,16):
    plt.annotate('power={}'.format(i), (0.5, 7))
    loss,weights,y_pred = polyRegress(x,y,i)
    print("For ",i, "th order the loss is ",loss)
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y_pred, color='red')
    plt.show()

print("After the 10th order the regression becomes worse")
