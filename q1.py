import numpy as np

def train(dataset,epoch):
    #initialise perceptron weight and bias
    weight = np.transpose(np.zeros(2))
    bias = np.zeros(1)
    #iterate for n epoch
    for j in range(epoch):
        for i in range(len(dataset)):
            #perceptron update rule
            if np.sign(np.matmul(weight , dataset[i][0:2])+bias) != np.sign(dataset[i][2]):
                #update weight
                weight[0] += dataset[i][0]*dataset[i][2]
                weight[1] += dataset[i][1]*dataset[i][2]
                #update offset
                bias += dataset[i][2]
    return (weight,bias)

def eval(test,weight,bias):
    print(test[1][2])
    print(np.matmul(weight, test[21][0:2]) + bias)
    correct = 0
    for i in range(len(test)):
        # perceptron update rule
        if (np.sign(np.matmul(weight, test[i][0:2])+bias)) == np.sign(test[i][2]):
            correct += 1
    return (correct/len(test))

def main():
    # read csv as dataframe
    dataset = np.genfromtxt('train_1_5.csv', delimiter=',')
    test = np.genfromtxt('test_1_5.csv', delimiter=',')

    # part (a)
    weight_1,bias_1 = train(dataset,1)
    accuracy = eval(test,weight_1,bias_1)
    print("For 1 epoch, the theta is ",weight_1," and the offset is ",bias_1, "with an accuracy of",accuracy)

    # part (b)
    weight_5, bias_5 = train(dataset, 5)
    accuracy = eval(test,weight_5,bias_5)
    print("For 5 epochs, the theta is ",weight_5," and the offset is ",bias_5, "with an accuracy of",accuracy)


if __name__ == "__main__":
    main()
