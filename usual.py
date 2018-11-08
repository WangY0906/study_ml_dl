#coding utf-8
import sys,os
import numpy as np
import pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])

print(x)

y = np.argmax(x, axis=1)

print(y)