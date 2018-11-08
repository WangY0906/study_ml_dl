import numpy as np
import os,sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import cross_entropy_error,softmax
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x,self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss



net = simpleNet()
#print(net.W)
x = np.array([0.6,0.9])
t = np.array([0,0,1])
def f(W):
    return net.loss(x,t)

dW = numerical_gradient(f, net.W)
print(dW)

x = np.array([0.6,0.9])
p = net.predict(x)
#print(p)
#print(np.argmax(p))

t = np.array([0,0,1])
#print(net.loss(x,t))

'''def softmax(a):
    p = np.max(a)
    exp_a = np.exp(a-p)
    return exp_a / np.sum(exp_a)

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7 #protect
    return -np.sum(t*np.log(y+delta))

def batch_cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def no_onehot_batch_cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+ 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        #caculate f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #caculate f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x = x - lr*grad
    return x


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    return x_test, t_test

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_train.shape)
print(t_train.shape)
'''