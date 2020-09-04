#from cusignal import chirp, spectrogram
import numpy as np
# Bandwidt
from numpy import random
#from LFMSignal import *
from scipy.signal import chirp, spectrogram, hilbert
import matplotlib.pyplot as plt

# Dataset consts
seconds = pw = 50e-6
b0, b1 = 4e6, 6e6-2e5
fs = 2 * 8e6
classes = 10
bandwidths = np.linspace(b0, b1, num=classes)

# Create labels and preallocate numpy array
"""

"""
dataset = np.zeros((int(1e5), 800, 2), dtype=np.float32)
labels = np.zeros((10000)) # 1e4 x 10 classes
for i in range(1,10):
	labels = np.concatenate((labels, np.ones((10000))*i))
labels = labels.astype(np.int64)

class_samples = 10000
# Iter classes
for label in range(classes):
	#number of samples per class
	for j in range(class_samples):

		idx = j + class_samples*label 
		#if idx % 1000 == 0: print("Sample: {} - {}%".format(idx, 100*idx/(class_samples*classes)), end=" ")
		seconds = pw = 50e-6
		fs = 2 * 8e6
		f0 = random.randint(0, 2e6)
		f1 = f0 + bandwidths[label]
		t = np.linspace(0, seconds, seconds * fs, endpoint=False)
		data = chirp(t, f0=f0, f1=f1, t1=seconds, method='linear')
		dataset[idx] = hilbert(data).view('(2,)float').astype('float32')

# Shuffle and save
indices = np.arange(dataset.shape[0])
np.random.shuffle(indices)
dataset = dataset[indices]
labels = labels[indices]

with open('complex_dataset_10class_train.npy', 'wb') as f:
    np.save(f, dataset[:80000])

with open('complex_dataset_10class_val.npy', 'wb') as f:
    np.save(f, dataset[80000:90000])

with open('complex_dataset_10class_test.npy', 'wb') as f:
    np.save(f, dataset[90000:])  

with open('complex_dataset_10class_train_labels.npy', 'wb') as f:
    np.save(f, labels[:80000])

with open('complex_dataset_10class_val_labels.npy', 'wb') as f:
    np.save(f, labels[80000:90000])

with open('complex_dataset_10class_test_labels.npy', 'wb') as f:
    np.save(f, labels[90000:])  




