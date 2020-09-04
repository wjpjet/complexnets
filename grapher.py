import matplotlib.pyplot as plt
import numpy as np 
import json

file_path = "real.out"

with open(file_path) as f:
    lines = [line.rstrip() for line in f if 'Val:' in line or "Train:" in line]

tr, val = [], []

for line in lines:
	
	num = line.split(" ")[3]
	if "Train:" in line:
		tr.append(float(num))
	else:
		val.append(float(num))

x = [v for v in range(len(tr))]
print(tr)
fig, axs = plt.subplots()
axs.plot(x, tr)
axs.plot(x, val)
axs.set(xlabel='epoch', ylabel='loss',
       title='Real')
axs.grid()
plt.show()
