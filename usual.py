import numpy as np
import matplotlib.pylab as plt

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def qiudao(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/   (2*h)
def function_2(x):
    return np.sum(x**2)

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


print(gradient_descent(function_2,np.array([-3.0,4.0]),lr=0.1,step_num=100))