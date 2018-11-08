import numpy as np
import matplotlib.pylab as plt
import os,sys
sys.path.append(os.pardir)
from common.multi_layer_net import MultiLayerNet
from common.util import smooth_curve
from dataset.mnist import load_mnist
from  common.optimizer import *

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

weight_init_types = {'std=0.01':0.01, 'Xavier':'sigmoid', 'He':'relu'}
optizer = SGD(lr=0.01)

network = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    network[key] = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100], output_size=10, weight_init_std=weight_type)
    train_loss[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():
        grads = network[key].gradient(x_batch, t_batch)
        optizer.update(network[key].params, grads)

        loss = network[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("========" + "iteration:" + str(i) + "========")
        for key in weight_init_types.keys():
            loss = network[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))



markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()