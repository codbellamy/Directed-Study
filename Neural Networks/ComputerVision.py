import matplotlib.pyplot as plt
import numpy as np
import pickle

LABELS = { 0: 'airplaine', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
           5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
i = 69

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
    return d

d = unpickle('./CIFAR10/data_batch_1')


imgs = np.array(d['data'])

imgs = imgs.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')

plt.imshow(imgs[i])
plt.show()

print(LABELS[d['labels'][i]])