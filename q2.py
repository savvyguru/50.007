from numpy.linalg import inv
import numpy as np
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

def CFLR(X,y):
    # calculate coefficients using closed-form solution
    coeffs = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return coeffs

def plot_LR(X,x,y,coeffs):
    print("plotting")
    #predict y
    y_pred = X.dot(coeffs)

    #plot graph
    plt.scatter(x, y)
    plt.plot(x, y_pred, color='red')
    plt.show()

def empirical_risk(y,y_pred):
    #calculate empirical risk
    diff = y-y_pred
    square_loss = (diff**2) / 2
    empirical_risk = sum(square_loss)/len(y)
    return empirical_risk

def BGD(X, y, theta, alpha, num_iters):
    m = y.size  # number of training examples
    for i in range(num_iters):
        y_hat = np.dot(X, theta)
        theta = theta - alpha * (1.0/m) * np.dot(np.transpose(X), y_hat - y)
    return theta

def SGD(X,y,theta, alpha):
    m = y.size  # number of training examples
    for j in range(m):
        rand_ind = np.random.randint(0,m)
        X_i = X[rand_ind, :].reshape(1, X.shape[1])
        y_i = y[rand_ind].reshape(1, 1)
        prediction = np.dot(X_i, theta)
        theta = theta + alpha*(X_i.T.dot((y_i-prediction)))
    return theta

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

def polyRegress(X,y,d=2):
    '''
    X: features (without bias), vector
    y: labels, vector
    d: order, greater than 0
    '''
    assert d>0
    X_poly = []
    X = np.array(X)
    #high order features
    for i in range(d+1):
        X_poly.append(X**i)
    X_poly = np.array(X_poly)
    #closed form regression
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    cost = cal_cost(theta, X_poly.T, y)
    return theta,X_poly, cost

def main():
    # X has column vector of ones while x does not
    X,x,y = load_data("hw1x.dat","hw1y.dat")
    coeffs = CFLR(X,y)
    print("The weight vector obtained from closed form linear regression is ",coeffs)
    plot_LR(X,x,y,coeffs)
    y_pred = X.dot(coeffs)
    risk = empirical_risk(y, y_pred)
    print("The training error is ",risk[0])

    # batch gradient descent
    bgd_theta = np.zeros((2,1))
    BGD_weight_vector = BGD(X,y,bgd_theta,0.01,5)
    print("Weight vector obtained from batch gradient descent is ",BGD_weight_vector)
    bgd_pred = X.dot(BGD_weight_vector)
    bgd_risk = empirical_risk(y,bgd_pred)
    print("Training error for batch gradient descent is ",bgd_risk)

    #stochastic gradient descent
    for i in range(5):
        sgd_theta = np.zeros((2, 1))
        SGD_weight_vector = SGD(X, y, sgd_theta, 0.01)
        print("Weight vector obtained from stochastic gradient descent interation ",i," is ", SGD_weight_vector)
        sgd_pred = X.dot(SGD_weight_vector)
        sgd_risk = empirical_risk(y, sgd_pred)
        print("Training error for batch gradient descent is ", sgd_risk)

    #polynomial regression
    thetas = []
    X_polys = []
    for d in range(1, 12):
        theta, X_poly, error = polyRegress(x, y, d)
        thetas.append(theta)
        X_polys.append(X_poly)
        print('d={}'.format(d), 'Error:{:.5f}'.format(error))
    plt.figure(figsize=(12, 8))

    for i, theta in enumerate(thetas[:9]):
        X_poly = X_polys[i]
        preds = X_poly.T.dot(theta)
        plt.subplot(3, 3, i + 1)
        plt.plot(X, y, '.')
        plt.plot(X, preds)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.annotate('d={}'.format(i + 1), (0.5, 7))

if __name__ == '__main__':
    main()