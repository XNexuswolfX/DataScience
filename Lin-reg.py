import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# To store column mean and std.
mu = []
sd = []
# Function to load data
def load_data(filename,no_of_features,colnames):
    df = pd.read_csv(filename,index_col=False)
    df.columns = colnames
    data = np.array(df,dtype=float)
    plot_data(data[:, :no_of_features], data[:, -1])
    normalize(data)
    x = data[:,:no_of_features]
    x = np.hstack((np.ones((x.shape[0],1)),x))
    y = data[:,-1].reshape(-1,1)
    return x,y

# Fucntion to plot data.
def plot_data(x, y):
    plt.xlabel('house size')
    plt.ylabel('price')
    plt.plot(x[:, 0], y, 'bo')
    plt.plot(x[:, 1], y, 'ro')
    plt.show()

# Function to normalize the data.
def normalize(data):
    for i in range(0,data.shape[1]-1):
        data[:,i] = (data[:,i] - np.mean(data[:,i]))/(np.std(data[:,i]))
        mu.append(np.mean(data[:,i]))
        sd.append(np.std(data[:,i]))

# Hypothesis of x and theta byb the model.
def h(x,theta):
    return np.matmul(x,theta)

# Cost Function of linear regression
def cost_function(x,y,theta):
    return ((h(x,theta)-y).T@(h(x,theta)-y))/(2 * y.shape[0])

# Obtaining optimal value of theta and cost function.
def gradient_descent(x,y,theta,learning_rate,epochs):
    m = x.shape[0]
    J_all = []

    for _ in range(epochs):
        h_x = h(x,theta)
        cost_der = (x.T @ (h_x-y)) / (m)
        theta = theta - (learning_rate) * cost_der
        J_all.append(cost_function(x,y,theta))
    return theta, J_all

# Plot the cost function change through epochs.
def plot_cost(J_all, num_epochs):
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(num_epochs, J_all, 'm', linewidth="5")
    plt.show()

# Function to predict values after model is trained.
def test(theta, x):
    x[0] = (x[0] - mu[0]) / sd[0]
    x[1] = (x[1] - mu[1]) / sd[1]

    y = theta[0] + theta[1] * x[0] + theta[2] * x[1]
    print("Price of house: ", y)

x, y = load_data("/Users/adityaratna/Desktop/Study/Statistical Machine Learning/Codes/DATA/house_price_data.txt",2,['hsize','rooms','price'])
# y = np.reshape(y, (46, 1))
# x = np.hstack((np.ones((x.shape[0], 1)), x))
theta = np.zeros((x.shape[1], 1))
learning_rate = 0.1
num_epochs = 50

theta_star, J_all_star = gradient_descent(x=x,y=y,theta=theta,learning_rate=learning_rate,epochs=num_epochs)
J = cost_function(x=x, y=y, theta=theta_star)

print("Cost: ", J)
print("Parameters: ", theta_star)

# for testing and plotting cost
n_epochs = []
jplot = []
count = 0
for i in J_all_star:
    jplot.append(i[0][0])
    n_epochs.append(count)
    count += 1
jplot = np.array(jplot)
n_epochs = np.array(n_epochs)
plot_cost(jplot, n_epochs)

test(theta=theta_star,x=[1600,3])

