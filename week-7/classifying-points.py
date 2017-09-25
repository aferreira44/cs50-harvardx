import matplotlib.pyplot as plt     # for generating plots
import numpy as np                  # for scientific computing

%matplotlib inline

# Nearest Neighbor Classifier

def dist(x, y):
    return np.sqrt(np.sum((x - y)**2))

X_train = np.array([[1,1], [2,2.5], [3,1.2], [5.5,6.3], [6,9], [7,6]])
Y_train = ['red', 'red', 'red', 'blue', 'blue', 'blue']

print("X_train[5,0]: " + str(X_train[5,0]))
print("X_train[5,1]: " + str(X_train[5,1]))

# slicing syntax

print(X_train[:,0])
print(X_train[:,1])

# plot the training set

plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], s = 170, color = Y_train[:])
plt.show()

# create a new test point

X_test = np.array([3,4])

# plot again

plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], s = 170, color = Y_train[:])
plt.scatter(X_test[0], X_test[1], s = 170, color = 'green')
plt.show()

# for each point in x_train we compute it distance to X_test

num = len(X_train)          # number of points in X_train
distance = np.zeros(num)    # numpy arrays of zeros
for i in range(num):
    distance[i] = dist(X_train[i], X_test)
print(distance)

# choose the point in x_train with the minimal distance from X_new

min_index = np.argmin(distance)    # Index with smallest distance
print(Y_train[min_index])