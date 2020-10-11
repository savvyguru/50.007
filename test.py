import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_df = pd.read_csv('hw1_ridge_x.dat', sep=",", header=None)
y_df = pd.read_csv('hw1_ridge_y.dat', sep=",", header=None)

vX = x_df.head(10)
vY = y_df.head(10)
tX = x_df.tail(40)
tY = y_df.tail(40)

def ridge_regression(tX, tY, l):
    n,d = tX.shape
    i = np.identity(d, dtype=float)
    i = pd.DataFrame(data=i)
    exp1 = (n*l*i + tX.T.dot(tX))
    exp1_inv = pd.DataFrame(np.linalg.pinv(exp1))
    exp2 = (tX.T).dot(tY)
    return exp1_inv.dot(exp2)

exact_sol = ridge_regression(tX, tY, 0.15)
print("The exact solution theta for ridge regression is ",exact_sol)

tn = tX.shape[0]
vn = vX.shape[0]
tloss = []
vloss = []
index = -np.arange(0,5,0.1)

for i in index:
    w = ridge_regression(tX,tY,10**i)
    tloss = tloss + [np.sum((np.dot(tX,w)-tY)**2)/tn/2]
    vloss = vloss + [np.sum((np.dot(vX,w)-vY)**2)/vn/2]

plt.plot(index,np.log(tloss),'r')
plt.plot(index,np.log(vloss),'b')
plt.show()

min_vloss = float('inf')
ind = 0
for i,l in enumerate(vloss):
    val = l.iloc[0]
    if val < min_vloss:
        min_vloss = val
        ind = i

best_lambda = 10**index[ind]
print("The value of lambda that minimises validation loss is ",best_lambda)